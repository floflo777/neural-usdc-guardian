const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

describe("NeuralUSDCGuardian", function () {
  let guardian, usdc;
  let owner, user1, user2, user3, guardian1, guardian2, guardian3, attacker;

  const USDC_DECIMALS = 6;
  const toUSDC = (n) => ethers.parseUnits(n.toString(), USDC_DECIMALS);
  const SCALE = ethers.parseEther("1"); // 1e18
  const ONE_HOUR = 3600;

  beforeEach(async function () {
    [owner, user1, user2, user3, guardian1, guardian2, guardian3, attacker] =
      await ethers.getSigners();

    const MockUSDC = await ethers.getContractFactory("MockUSDC");
    usdc = await MockUSDC.deploy();

    const Guardian = await ethers.getContractFactory("NeuralUSDCGuardian");
    guardian = await Guardian.deploy(await usdc.getAddress());

    // Setup guardians
    await guardian.addGuardian(guardian1.address);
    await guardian.addGuardian(guardian2.address);
    await guardian.addGuardian(guardian3.address);

    // Mint USDC to test users
    for (const u of [user1, user2, user3]) {
      await usdc.mint(u.address, toUSDC(100000));
      await usdc
        .connect(u)
        .approve(await guardian.getAddress(), ethers.MaxUint256);
    }
  });

  // ================================================================
  //  DEPLOYMENT
  // ================================================================

  describe("Deployment", function () {
    it("Should set the USDC address", async function () {
      expect(await guardian.usdc()).to.equal(await usdc.getAddress());
    });

    it("Should set the deployer as owner", async function () {
      expect(await guardian.owner()).to.equal(owner.address);
    });

    it("Should initialise default thresholds", async function () {
      expect(await guardian.lowThreshold()).to.equal(30);
      expect(await guardian.highThreshold()).to.equal(70);
      expect(await guardian.timelockDuration()).to.equal(ONE_HOUR);
      expect(await guardian.minApprovals()).to.equal(2);
    });

    it("Should initialise model weights", async function () {
      // Neuron 0 amountRatio weight = 2 * SCALE
      expect(await guardian.weightsHidden(0)).to.equal(SCALE * 2n);
      // Output bias = -0.30
      expect(await guardian.biasOutput()).to.equal((SCALE * -3n) / 10n);
      // Trust safety net weight = -0.70
      expect(await guardian.weightsOutput(6)).to.equal((SCALE * -70n) / 100n);
    });

    it("Should revert with zero USDC address", async function () {
      const G = await ethers.getContractFactory("NeuralUSDCGuardian");
      await expect(G.deploy(ethers.ZeroAddress)).to.be.revertedWithCustomError(
        guardian,
        "ZeroAddress"
      );
    });
  });

  // ================================================================
  //  REGISTRATION
  // ================================================================

  describe("Registration", function () {
    it("Should register a new user", async function () {
      await expect(guardian.connect(user1).register())
        .to.emit(guardian, "Registered")
        .withArgs(user1.address);

      const p = await guardian.getProfile(user1.address);
      expect(p.isRegistered).to.be.true;
    });

    it("Should revert on double registration", async function () {
      await guardian.connect(user1).register();
      await expect(
        guardian.connect(user1).register()
      ).to.be.revertedWithCustomError(guardian, "AlreadyRegistered");
    });
  });

  // ================================================================
  //  DEPOSITS & WITHDRAWALS
  // ================================================================

  describe("Deposits & Withdrawals", function () {
    beforeEach(async function () {
      await guardian.connect(user1).register();
    });

    it("Should deposit USDC", async function () {
      await expect(guardian.connect(user1).deposit(toUSDC(1000)))
        .to.emit(guardian, "Deposited")
        .withArgs(user1.address, toUSDC(1000));

      expect(await guardian.balances(user1.address)).to.equal(toUSDC(1000));
    });

    it("Should revert deposit of 0", async function () {
      await expect(
        guardian.connect(user1).deposit(0)
      ).to.be.revertedWithCustomError(guardian, "ZeroAmount");
    });

    it("Should revert deposit without registration", async function () {
      await expect(
        guardian.connect(user2).deposit(toUSDC(100))
      ).to.be.revertedWithCustomError(guardian, "NotRegistered");
    });

    it("Should withdraw USDC", async function () {
      await guardian.connect(user1).deposit(toUSDC(1000));

      const balBefore = await usdc.balanceOf(user1.address);
      await expect(guardian.connect(user1).withdraw(toUSDC(500)))
        .to.emit(guardian, "Withdrawn")
        .withArgs(user1.address, toUSDC(500));

      expect(await guardian.balances(user1.address)).to.equal(toUSDC(500));
      expect(await usdc.balanceOf(user1.address)).to.equal(
        balBefore + toUSDC(500)
      );
    });

    it("Should revert withdrawal exceeding balance", async function () {
      await guardian.connect(user1).deposit(toUSDC(100));
      await expect(guardian.connect(user1).withdraw(toUSDC(200)))
        .to.be.revertedWithCustomError(guardian, "InsufficientBalance")
        .withArgs(toUSDC(100), toUSDC(200));
    });

    it("Should revert withdrawal of 0", async function () {
      await expect(
        guardian.connect(user1).withdraw(0)
      ).to.be.revertedWithCustomError(guardian, "ZeroAmount");
    });
  });

  // ================================================================
  //  NEURAL NETWORK - PURE FORWARD PASS
  // ================================================================

  describe("Neural Network", function () {
    describe("computeRisk (direct feature input)", function () {
      it("Normal trusted tx -> low risk (~18)", async function () {
        // features: [1.0, 0.2, 1.0, 0.3]
        const features = [
          SCALE, // amountRatio = 1.0
          (SCALE * 2n) / 10n, // txFreq = 0.2
          SCALE, // trusted
          (SCALE * 3n) / 10n, // recency = 0.3
        ];
        const risk = await guardian.computeRisk(features);
        expect(risk).to.be.lte(25);
        expect(risk).to.be.gte(10);
      });

      it("First tx to untrusted recipient -> medium risk (~56-66)", async function () {
        // features: [2.0, 0.0, 0.0, 0.0]
        const features = [
          SCALE * 2n, // default amountRatio for new user
          0n, // no recent tx
          0n, // untrusted
          0n, // not recent
        ];
        const risk = await guardian.computeRisk(features);
        expect(risk).to.be.gte(50);
        expect(risk).to.be.lte(75);
      });

      it("Large amount to untrusted -> high risk (~85+)", async function () {
        // features: [5.0, 0.0, 0.0, 0.0]
        const features = [
          SCALE * 5n,
          0n,
          0n,
          0n,
        ];
        const risk = await guardian.computeRisk(features);
        expect(risk).to.be.gte(80);
      });

      it("Dangerous: 10x amount, high freq, untrusted, very recent -> ~95+", async function () {
        // features: [10.0, 2.0, 0.0, 0.95]
        const features = [
          SCALE * 10n,
          SCALE * 2n,
          0n,
          (SCALE * 95n) / 100n,
        ];
        const risk = await guardian.computeRisk(features);
        expect(risk).to.be.gte(90);
      });

      it("Trusted recipient with high amount -> lower risk (~21-31)", async function () {
        // features: [3.0, 0.2, 1.0, 0.3]
        const features = [
          SCALE * 3n,
          (SCALE * 2n) / 10n,
          SCALE, // trusted
          (SCALE * 3n) / 10n,
        ];
        const risk = await guardian.computeRisk(features);
        expect(risk).to.be.lte(40);
      });

      it("Zero features -> moderate risk (no trust)", async function () {
        const features = [0n, 0n, 0n, 0n];
        const risk = await guardian.computeRisk(features);
        // Only untrusted-recipient alarm fires (bias = 2*SCALE)
        expect(risk).to.be.gte(30);
        expect(risk).to.be.lte(70);
      });

      it("All max features (untrusted) -> very high risk", async function () {
        const features = [
          SCALE * 10n, // max ratio
          SCALE * 2n, // max freq
          0n, // untrusted
          SCALE, // max recency
        ];
        const risk = await guardian.computeRisk(features);
        expect(risk).to.be.gte(93);
      });
    });

    describe("Sigmoid function behaviour", function () {
      it("Symmetric: extreme positive -> ~100, extreme negative -> ~0", async function () {
        const highRisk = await guardian.computeRisk([
          SCALE * 10n,
          SCALE * 2n,
          0n,
          SCALE,
        ]);
        const lowRisk = await guardian.computeRisk([
          SCALE,
          0n,
          SCALE,
          0n,
        ]);
        expect(highRisk).to.be.gt(lowRisk);
        expect(highRisk).to.be.gte(90);
        expect(lowRisk).to.be.lte(25);
      });
    });
  });

  // ================================================================
  //  TRANSFER FLOWS
  // ================================================================

  describe("Transfers", function () {
    beforeEach(async function () {
      await guardian.connect(user1).register();
      await guardian.connect(user2).register();
      await guardian.connect(user1).deposit(toUSDC(50000));
    });

    describe("Input validation", function () {
      it("Should revert transfer of 0", async function () {
        await expect(
          guardian.connect(user1).transfer(user2.address, 0)
        ).to.be.revertedWithCustomError(guardian, "ZeroAmount");
      });

      it("Should revert transfer to zero address", async function () {
        await expect(
          guardian.connect(user1).transfer(ethers.ZeroAddress, toUSDC(10))
        ).to.be.revertedWithCustomError(guardian, "ZeroAddress");
      });

      it("Should revert transfer to self", async function () {
        await expect(
          guardian.connect(user1).transfer(user1.address, toUSDC(10))
        ).to.be.revertedWithCustomError(guardian, "SelfTransferNotAllowed");
      });

      it("Should revert if not registered", async function () {
        await expect(
          guardian.connect(user3).transfer(user1.address, toUSDC(10))
        ).to.be.revertedWithCustomError(guardian, "NotRegistered");
      });

      it("Should revert if insufficient balance", async function () {
        await expect(
          guardian.connect(user1).transfer(user2.address, toUSDC(999999))
        ).to.be.revertedWithCustomError(guardian, "InsufficientBalance");
      });
    });

    describe("Low risk (auto-approve)", function () {
      it("Should auto-approve transfer to trusted recipient", async function () {
        await guardian
          .connect(user1)
          .setTrustedRecipient(user2.address, true);

        // Wait for enough time so recency = 0
        await time.increase(ONE_HOUR + 1);

        const tx = await guardian
          .connect(user1)
          .transfer(user2.address, toUSDC(100));
        const receipt = await tx.wait();

        // Should emit TransferAutoApproved
        const autoEvent = receipt.logs.find((l) => {
          try {
            return guardian.interface.parseLog(l)?.name === "TransferAutoApproved";
          } catch { return false; }
        });
        expect(autoEvent).to.not.be.undefined;

        // Recipient balance updated
        expect(await guardian.balances(user2.address)).to.equal(toUSDC(100));
      });

      it("Should emit RiskAssessed with features", async function () {
        await guardian
          .connect(user1)
          .setTrustedRecipient(user2.address, true);
        await time.increase(ONE_HOUR + 1);

        await expect(
          guardian.connect(user1).transfer(user2.address, toUSDC(100))
        ).to.emit(guardian, "RiskAssessed");
      });
    });

    describe("Medium risk (timelock)", function () {
      it("Should create pending tx for untrusted recipient", async function () {
        // First tx to untrusted recipient => medium risk
        await time.increase(ONE_HOUR + 1);

        const tx = await guardian
          .connect(user1)
          .transfer(user2.address, toUSDC(100));
        const receipt = await tx.wait();

        const pendingEvent = receipt.logs.find((l) => {
          try {
            return guardian.interface.parseLog(l)?.name === "TransferPending";
          } catch { return false; }
        });
        expect(pendingEvent).to.not.be.undefined;

        const parsed = guardian.interface.parseLog(pendingEvent);
        const txId = parsed.args.txId;
        const riskScore = parsed.args.riskScore;

        // Should be in medium range
        expect(riskScore).to.be.gte(30);
        expect(riskScore).to.be.lt(70);

        // Pending tx stored
        const ptx = await guardian.getPendingTx(txId);
        expect(ptx.sender).to.equal(user1.address);
        expect(ptx.recipient).to.equal(user2.address);
        expect(ptx.amount).to.equal(toUSDC(100));
        expect(ptx.executed).to.be.false;
        expect(ptx.cancelled).to.be.false;
      });

      it("Should lock sender funds in pending tx", async function () {
        await time.increase(ONE_HOUR + 1);
        const balBefore = await guardian.balances(user1.address);

        await guardian.connect(user1).transfer(user2.address, toUSDC(100));

        expect(await guardian.balances(user1.address)).to.equal(
          balBefore - toUSDC(100)
        );
      });
    });

    describe("High risk (multi-sig)", function () {
      it("Should require multi-sig for very large transfers to untrusted", async function () {
        // Build some history with small amounts so large amount is suspicious
        await guardian
          .connect(user1)
          .setTrustedRecipient(user2.address, true);
        for (let i = 0; i < 5; i++) {
          await time.increase(ONE_HOUR + 1);
          await guardian.connect(user1).transfer(user2.address, toUSDC(10));
        }

        // Now send 500 USDC (50x avg) to untrusted user3
        await time.increase(ONE_HOUR + 1);
        const tx = await guardian
          .connect(user1)
          .transfer(user3.address, toUSDC(500));
        const receipt = await tx.wait();

        const pendingEvent = receipt.logs.find((l) => {
          try {
            return guardian.interface.parseLog(l)?.name === "TransferPending";
          } catch { return false; }
        });
        expect(pendingEvent).to.not.be.undefined;

        const parsed = guardian.interface.parseLog(pendingEvent);
        expect(parsed.args.riskScore).to.be.gte(70);
        expect(parsed.args.requiresMultiSig).to.be.true;
      });
    });
  });

  // ================================================================
  //  PENDING TRANSACTIONS
  // ================================================================

  describe("Pending Transactions", function () {
    let txId;

    beforeEach(async function () {
      await guardian.connect(user1).register();
      await guardian.connect(user2).register();
      await guardian.connect(user1).deposit(toUSDC(50000));
      await time.increase(ONE_HOUR + 1);

      // Create a medium-risk pending tx (untrusted recipient)
      const tx = await guardian
        .connect(user1)
        .transfer(user2.address, toUSDC(100));
      const receipt = await tx.wait();
      const ev = receipt.logs.find((l) => {
        try {
          return guardian.interface.parseLog(l)?.name === "TransferPending";
        } catch { return false; }
      });
      txId = guardian.interface.parseLog(ev).args.txId;
    });

    describe("Timelock execution", function () {
      it("Should revert before timelock expires", async function () {
        await expect(
          guardian.executePending(txId)
        ).to.be.revertedWithCustomError(guardian, "TimelockNotExpired");
      });

      it("Should execute after timelock expires", async function () {
        await time.increase(ONE_HOUR + 1);

        await expect(guardian.executePending(txId))
          .to.emit(guardian, "TransferExecuted")
          .withArgs(txId);

        const ptx = await guardian.getPendingTx(txId);
        expect(ptx.executed).to.be.true;

        // Recipient received funds
        expect(await guardian.balances(user2.address)).to.equal(toUSDC(100));
      });

      it("Should revert double execution", async function () {
        await time.increase(ONE_HOUR + 1);
        await guardian.executePending(txId);

        await expect(
          guardian.executePending(txId)
        ).to.be.revertedWithCustomError(guardian, "TxAlreadyExecuted");
      });
    });

    describe("Cancellation", function () {
      it("Should allow sender to cancel", async function () {
        const balBefore = await guardian.balances(user1.address);

        await expect(guardian.connect(user1).cancelPending(txId))
          .to.emit(guardian, "TransferCancelled")
          .withArgs(txId);

        // Funds refunded
        expect(await guardian.balances(user1.address)).to.equal(
          balBefore + toUSDC(100)
        );

        const ptx = await guardian.getPendingTx(txId);
        expect(ptx.cancelled).to.be.true;
      });

      it("Should revert cancel by non-sender", async function () {
        await expect(
          guardian.connect(user2).cancelPending(txId)
        ).to.be.revertedWithCustomError(guardian, "NotPendingTxSender");
      });

      it("Should revert execute after cancel", async function () {
        await guardian.connect(user1).cancelPending(txId);
        await time.increase(ONE_HOUR + 1);

        await expect(
          guardian.executePending(txId)
        ).to.be.revertedWithCustomError(guardian, "TxAlreadyCancelled");
      });

      it("Should revert double cancel", async function () {
        await guardian.connect(user1).cancelPending(txId);
        await expect(
          guardian.connect(user1).cancelPending(txId)
        ).to.be.revertedWithCustomError(guardian, "TxAlreadyCancelled");
      });
    });

    describe("Guardian multi-sig approval", function () {
      let highRiskTxId;

      beforeEach(async function () {
        // Build history with small trusted transfers
        await guardian
          .connect(user1)
          .setTrustedRecipient(user2.address, true);
        for (let i = 0; i < 5; i++) {
          await time.increase(ONE_HOUR + 1);
          await guardian.connect(user1).transfer(user2.address, toUSDC(10));
        }

        // Large transfer to untrusted user3 => high risk
        await time.increase(ONE_HOUR + 1);
        const tx = await guardian
          .connect(user1)
          .transfer(user3.address, toUSDC(500));
        const receipt = await tx.wait();
        const ev = receipt.logs.find((l) => {
          try {
            const parsed = guardian.interface.parseLog(l);
            return parsed?.name === "TransferPending" && parsed.args.requiresMultiSig;
          } catch { return false; }
        });

        if (ev) {
          highRiskTxId = guardian.interface.parseLog(ev).args.txId;
        } else {
          // Fallback: use the last pending tx
          highRiskTxId = (await guardian.pendingTxCount()) - 1n;
        }
      });

      it("Should allow guardians to approve", async function () {
        await expect(guardian.connect(guardian1).approvePending(highRiskTxId))
          .to.emit(guardian, "GuardianApproval")
          .withArgs(highRiskTxId, guardian1.address);

        const ptx = await guardian.getPendingTx(highRiskTxId);
        expect(ptx.approvalCount).to.equal(1);
      });

      it("Should revert approval by non-guardian", async function () {
        await expect(
          guardian.connect(attacker).approvePending(highRiskTxId)
        ).to.be.revertedWithCustomError(guardian, "NotGuardian");
      });

      it("Should revert duplicate approval", async function () {
        await guardian.connect(guardian1).approvePending(highRiskTxId);
        await expect(
          guardian.connect(guardian1).approvePending(highRiskTxId)
        ).to.be.revertedWithCustomError(guardian, "AlreadyApproved");
      });

      it("Should execute after sufficient approvals", async function () {
        await guardian.connect(guardian1).approvePending(highRiskTxId);
        await guardian.connect(guardian2).approvePending(highRiskTxId);

        await expect(guardian.executePending(highRiskTxId))
          .to.emit(guardian, "TransferExecuted")
          .withArgs(highRiskTxId);
      });

      it("Should revert execute with insufficient approvals", async function () {
        await guardian.connect(guardian1).approvePending(highRiskTxId);
        // Only 1 approval, need 2

        await expect(
          guardian.executePending(highRiskTxId)
        ).to.be.revertedWithCustomError(guardian, "InsufficientApprovals");
      });
    });
  });

  // ================================================================
  //  TRUSTED RECIPIENTS
  // ================================================================

  describe("Trusted Recipients", function () {
    it("Should set trusted recipient", async function () {
      await expect(
        guardian.connect(user1).setTrustedRecipient(user2.address, true)
      )
        .to.emit(guardian, "TrustedRecipientSet")
        .withArgs(user1.address, user2.address, true);

      expect(
        await guardian.trustedRecipients(user1.address, user2.address)
      ).to.be.true;
    });

    it("Should revoke trust", async function () {
      await guardian.connect(user1).setTrustedRecipient(user2.address, true);
      await guardian.connect(user1).setTrustedRecipient(user2.address, false);

      expect(
        await guardian.trustedRecipients(user1.address, user2.address)
      ).to.be.false;
    });

    it("Should revert with zero address", async function () {
      await expect(
        guardian.connect(user1).setTrustedRecipient(ethers.ZeroAddress, true)
      ).to.be.revertedWithCustomError(guardian, "ZeroAddress");
    });
  });

  // ================================================================
  //  ADMIN: GUARDIANS
  // ================================================================

  describe("Admin - Guardians", function () {
    it("Should add a guardian", async function () {
      const newG = attacker; // reusing signer
      await expect(guardian.addGuardian(newG.address))
        .to.emit(guardian, "GuardianAdded")
        .withArgs(newG.address);

      expect(await guardian.guardians(newG.address)).to.be.true;
      expect(await guardian.guardianCount()).to.equal(4);
    });

    it("Should revert adding duplicate guardian", async function () {
      await expect(
        guardian.addGuardian(guardian1.address)
      ).to.be.revertedWithCustomError(guardian, "AlreadyGuardian");
    });

    it("Should revert adding zero address guardian", async function () {
      await expect(
        guardian.addGuardian(ethers.ZeroAddress)
      ).to.be.revertedWithCustomError(guardian, "ZeroAddress");
    });

    it("Should remove a guardian", async function () {
      await expect(guardian.removeGuardian(guardian3.address))
        .to.emit(guardian, "GuardianRemoved")
        .withArgs(guardian3.address);

      expect(await guardian.guardians(guardian3.address)).to.be.false;
      expect(await guardian.guardianCount()).to.equal(2);
    });

    it("Should revert removing non-guardian", async function () {
      await expect(
        guardian.removeGuardian(attacker.address)
      ).to.be.revertedWithCustomError(guardian, "NotGuardian");
    });

    it("Should revert removing when at minimum", async function () {
      // Remove until at minApprovals (2), then try to remove one more
      await guardian.removeGuardian(guardian3.address); // 3 -> 2
      await expect(
        guardian.removeGuardian(guardian2.address)
      ).to.be.revertedWithCustomError(guardian, "CannotRemoveLastGuardian");
    });

    it("Should restrict guardian management to owner", async function () {
      await expect(
        guardian.connect(user1).addGuardian(attacker.address)
      ).to.be.revertedWithCustomError(guardian, "OwnableUnauthorizedAccount");
    });
  });

  // ================================================================
  //  ADMIN: MODEL UPDATES
  // ================================================================

  describe("Admin - Model Updates", function () {
    it("Should update model weights", async function () {
      const wH = new Array(32).fill(SCALE);
      const bH = new Array(8).fill(0n);
      const wO = new Array(8).fill(SCALE / 10n);
      const bO = 0n;

      await expect(guardian.updateModel(wH, bH, wO, bO)).to.emit(
        guardian,
        "ModelUpdated"
      );

      expect(await guardian.weightsHidden(0)).to.equal(SCALE);
    });

    it("Should enforce cooldown between updates", async function () {
      const wH = new Array(32).fill(0n);
      const bH = new Array(8).fill(0n);
      const wO = new Array(8).fill(0n);
      const bO = 0n;

      await guardian.updateModel(wH, bH, wO, bO);

      await expect(
        guardian.updateModel(wH, bH, wO, bO)
      ).to.be.revertedWithCustomError(guardian, "ModelUpdateTooFrequent");
    });

    it("Should allow update after cooldown", async function () {
      const wH = new Array(32).fill(0n);
      const bH = new Array(8).fill(0n);
      const wO = new Array(8).fill(0n);
      const bO = 0n;

      await guardian.updateModel(wH, bH, wO, bO);
      await time.increase(ONE_HOUR + 1);
      await expect(guardian.updateModel(wH, bH, wO, bO)).to.not.be.reverted;
    });

    it("Should restrict model updates to owner", async function () {
      const wH = new Array(32).fill(0n);
      const bH = new Array(8).fill(0n);
      const wO = new Array(8).fill(0n);
      const bO = 0n;

      await expect(
        guardian.connect(user1).updateModel(wH, bH, wO, bO)
      ).to.be.revertedWithCustomError(guardian, "OwnableUnauthorizedAccount");
    });
  });

  // ================================================================
  //  ADMIN: CONFIGURATION
  // ================================================================

  describe("Admin - Configuration", function () {
    it("Should update thresholds", async function () {
      await expect(guardian.setThresholds(20, 80))
        .to.emit(guardian, "ThresholdsUpdated")
        .withArgs(20, 80);

      expect(await guardian.lowThreshold()).to.equal(20);
      expect(await guardian.highThreshold()).to.equal(80);
    });

    it("Should revert invalid thresholds (low >= high)", async function () {
      await expect(
        guardian.setThresholds(70, 70)
      ).to.be.revertedWithCustomError(guardian, "InvalidThresholds");
    });

    it("Should revert thresholds > 100", async function () {
      await expect(
        guardian.setThresholds(30, 101)
      ).to.be.revertedWithCustomError(guardian, "InvalidThresholds");
    });

    it("Should update timelock duration", async function () {
      await expect(guardian.setTimelockDuration(7200))
        .to.emit(guardian, "TimelockDurationUpdated")
        .withArgs(7200);

      expect(await guardian.timelockDuration()).to.equal(7200);
    });

    it("Should update minApprovals", async function () {
      await expect(guardian.setMinApprovals(3))
        .to.emit(guardian, "MinApprovalsUpdated")
        .withArgs(3);

      expect(await guardian.minApprovals()).to.equal(3);
    });

    it("Should revert minApprovals of 0", async function () {
      await expect(
        guardian.setMinApprovals(0)
      ).to.be.revertedWithCustomError(guardian, "ZeroAmount");
    });
  });

  // ================================================================
  //  SECURITY: PAUSABLE
  // ================================================================

  describe("Security - Pausable", function () {
    beforeEach(async function () {
      await guardian.connect(user1).register();
      await guardian.connect(user1).deposit(toUSDC(1000));
    });

    it("Should pause deposits", async function () {
      await guardian.pause();
      await expect(
        guardian.connect(user1).deposit(toUSDC(100))
      ).to.be.revertedWithCustomError(guardian, "EnforcedPause");
    });

    it("Should pause transfers", async function () {
      await guardian.pause();
      await expect(
        guardian.connect(user1).transfer(user2.address, toUSDC(10))
      ).to.be.revertedWithCustomError(guardian, "EnforcedPause");
    });

    it("Should pause withdrawals", async function () {
      await guardian.pause();
      await expect(
        guardian.connect(user1).withdraw(toUSDC(10))
      ).to.be.revertedWithCustomError(guardian, "EnforcedPause");
    });

    it("Should unpause and resume operations", async function () {
      await guardian.pause();
      await guardian.unpause();
      await expect(guardian.connect(user1).deposit(toUSDC(100))).to.not.be
        .reverted;
    });

    it("Should restrict pause/unpause to owner", async function () {
      await expect(
        guardian.connect(user1).pause()
      ).to.be.revertedWithCustomError(guardian, "OwnableUnauthorizedAccount");
    });
  });

  // ================================================================
  //  VIEW HELPERS
  // ================================================================

  describe("View Helpers", function () {
    it("Should return full model weights", async function () {
      const [wH, bH, wO, bO] = await guardian.getModelWeights();
      expect(wH.length).to.equal(32);
      expect(bH.length).to.equal(8);
      expect(wO.length).to.equal(8);
      expect(wH[0]).to.equal(SCALE * 2n);
      expect(bO).to.equal((SCALE * -3n) / 10n);
    });

    it("assessRisk should return score and features", async function () {
      await guardian.connect(user1).register();
      const [risk, features] = await guardian.assessRisk(
        user1.address,
        user2.address,
        toUSDC(100)
      );
      expect(risk).to.be.gte(0);
      expect(risk).to.be.lte(100);
      expect(features.length).to.equal(4);
    });
  });

  // ================================================================
  //  EDGE CASES
  // ================================================================

  describe("Edge Cases", function () {
    beforeEach(async function () {
      await guardian.connect(user1).register();
      await guardian.connect(user2).register();
      await guardian.connect(user1).deposit(toUSDC(50000));
    });

    it("Should handle tx to non-existent pending ID", async function () {
      await expect(
        guardian.executePending(999)
      ).to.be.revertedWithCustomError(guardian, "TxDoesNotExist");
    });

    it("Should handle approval on non-existent pending ID", async function () {
      await expect(
        guardian.connect(guardian1).approvePending(999)
      ).to.be.revertedWithCustomError(guardian, "TxDoesNotExist");
    });

    it("Unregistered user can still receive and withdraw", async function () {
      // user3 is not registered but can receive via internal transfer
      await guardian
        .connect(user1)
        .setTrustedRecipient(user3.address, true);
      await time.increase(ONE_HOUR + 1);

      await guardian.connect(user1).transfer(user3.address, toUSDC(50));

      // user3 can withdraw without registering
      expect(await guardian.balances(user3.address)).to.equal(toUSDC(50));
      await guardian.connect(user3).withdraw(toUSDC(50));
      expect(await guardian.balances(user3.address)).to.equal(0);
    });

    it("Profile updates correctly over multiple transactions", async function () {
      await guardian
        .connect(user1)
        .setTrustedRecipient(user2.address, true);
      await time.increase(ONE_HOUR + 1);

      // 3 transactions
      await guardian.connect(user1).transfer(user2.address, toUSDC(100));
      await guardian.connect(user1).transfer(user2.address, toUSDC(200));
      await guardian.connect(user1).transfer(user2.address, toUSDC(300));

      const profile = await guardian.getProfile(user1.address);
      expect(profile.totalTransactions).to.equal(3);
      expect(profile.totalVolume).to.equal(toUSDC(600));
      expect(profile.recentTxCount).to.equal(3);
    });

    it("Recent window resets after 1 hour", async function () {
      await guardian
        .connect(user1)
        .setTrustedRecipient(user2.address, true);
      await time.increase(ONE_HOUR + 1);

      await guardian.connect(user1).transfer(user2.address, toUSDC(100));
      await guardian.connect(user1).transfer(user2.address, toUSDC(100));

      let profile = await guardian.getProfile(user1.address);
      expect(profile.recentTxCount).to.equal(2);

      // Wait > 1 hour
      await time.increase(ONE_HOUR + 1);
      await guardian.connect(user1).transfer(user2.address, toUSDC(100));

      profile = await guardian.getProfile(user1.address);
      expect(profile.recentTxCount).to.equal(1); // reset
      expect(profile.totalTransactions).to.equal(3); // cumulative
    });
  });
});
