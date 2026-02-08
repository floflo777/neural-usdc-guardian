// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * @title NeuralUSDCGuardian
 * @notice On-chain neural network that evaluates USDC transfer risk in real-time.
 * @dev Implements a 3-layer feedforward neural network (4-8-1) directly in Solidity
 *      using fixed-point arithmetic (18 decimal scale). Transactions are classified
 *      into low / medium / high risk and routed through the appropriate security gate.
 *
 * Architecture
 * ============
 *   Input Layer  (4 features)
 *     0 - amountRatio   : tx amount / sender's historical average
 *     1 - txFrequency   : recent tx count in rolling 1-hour window, normalised
 *     2 - recipientTrust: 1.0 if sender has marked recipient trusted, else 0.0
 *     3 - recency       : 1.0 if last tx was seconds ago, 0.0 if >= 1 hour ago
 *
 *   Hidden Layer (8 neurons, ReLU activation)
 *     Specialised pattern detectors for anomalous behaviour
 *
 *   Output Layer (1 neuron, fast-sigmoid activation)
 *     Risk score mapped to [0, 100]
 *
 * Risk Routing
 * ============
 *   [0,  lowThreshold)            -> Auto-approved (immediate transfer)
 *   [lowThreshold, highThreshold) -> Time-locked   (default 1 hour)
 *   [highThreshold, 100]          -> Multi-sig     (guardian approval required)
 */
contract NeuralUSDCGuardian is Ownable, ReentrancyGuard, Pausable {
    using SafeERC20 for IERC20;

    // ================================================================
    //                          CONSTANTS
    // ================================================================

    /// @dev Fixed-point scale factor (18 decimals)
    int256 public constant SCALE = 1e18;

    /// @dev Number of input features
    uint256 public constant INPUT_SIZE = 4;

    /// @dev Number of hidden-layer neurons
    uint256 public constant HIDDEN_SIZE = 8;

    /// @dev Maximum representable risk score
    uint256 public constant MAX_RISK = 100;

    /// @dev Minimum cooldown between model-weight updates
    uint256 public constant MODEL_UPDATE_COOLDOWN = 1 hours;

    /// @dev Cap on the amount-ratio feature to bound NN input
    int256 public constant MAX_AMOUNT_RATIO = 10e18; // 10.0

    /// @dev Cap on the frequency feature
    int256 public constant MAX_FREQUENCY = 2e18; // 2.0

    // ================================================================
    //                       CONFIGURATION
    // ================================================================

    /// @notice USDC token contract (immutable after deployment)
    IERC20 public immutable usdc;

    /// @notice Risk score below which transfers auto-approve
    uint256 public lowThreshold = 30;

    /// @notice Risk score at or above which transfers require multi-sig
    uint256 public highThreshold = 70;

    /// @notice Duration of the timelock for medium-risk transfers
    uint256 public timelockDuration = 1 hours;

    /// @notice Minimum guardian approvals for high-risk transfers
    uint256 public minApprovals = 2;

    /// @notice Timestamp of the last model-weight update
    uint256 public lastModelUpdate;

    // ================================================================
    //                    NEURAL NETWORK WEIGHTS
    // ================================================================

    /// @notice Hidden-layer weights, flattened [neuron * INPUT_SIZE + input]
    int256[32] public weightsHidden;

    /// @notice Hidden-layer biases
    int256[8] public biasesHidden;

    /// @notice Output-layer weights
    int256[8] public weightsOutput;

    /// @notice Output-layer bias
    int256 public biasOutput;

    // ================================================================
    //                        USER DATA
    // ================================================================

    /// @notice Behavioural profile built from transaction history
    struct UserProfile {
        uint256 totalTransactions;
        uint256 totalVolume;
        uint256 lastTransactionTime;
        uint256 recentTxCount;
        uint256 recentWindowStart;
        bool isRegistered;
    }

    /// @notice Address -> profile
    mapping(address => UserProfile) public profiles;

    /// @notice USDC balances held inside the guardian vault
    mapping(address => uint256) public balances;

    /// @notice Per-user trusted-recipient whitelist
    mapping(address => mapping(address => bool)) public trustedRecipients;

    // ================================================================
    //                         GUARDIANS
    // ================================================================

    /// @notice Whether an address is a guardian
    mapping(address => bool) public guardians;

    /// @notice Current number of active guardians
    uint256 public guardianCount;

    // ================================================================
    //                    PENDING TRANSACTIONS
    // ================================================================

    /// @notice A transfer that is awaiting timelock expiry or multi-sig approval
    struct PendingTx {
        address sender;
        address recipient;
        uint256 amount;
        uint256 riskScore;
        uint256 createdAt;
        uint256 approvalCount;
        bool executed;
        bool cancelled;
    }

    /// @notice txId -> pending transaction
    mapping(uint256 => PendingTx) public pendingTxs;

    /// @notice txId -> guardian address -> has approved
    mapping(uint256 => mapping(address => bool)) public hasApproved;

    /// @notice Auto-incrementing pending-transaction counter
    uint256 public pendingTxCount;

    // ================================================================
    //                       CUSTOM ERRORS
    // ================================================================

    error ZeroAmount();
    error ZeroAddress();
    error InsufficientBalance(uint256 available, uint256 required);
    error NotRegistered();
    error AlreadyRegistered();
    error NotGuardian();
    error AlreadyGuardian();
    error NotPendingTxSender();
    error TxAlreadyExecuted();
    error TxAlreadyCancelled();
    error TxDoesNotExist();
    error TimelockNotExpired(uint256 readyAt);
    error InsufficientApprovals(uint256 current, uint256 required);
    error AlreadyApproved();
    error ModelUpdateTooFrequent(uint256 nextAllowedAt);
    error InvalidThresholds();
    error CannotRemoveLastGuardian();
    error SelfTransferNotAllowed();

    // ================================================================
    //                          EVENTS
    // ================================================================

    event Registered(address indexed user);
    event Deposited(address indexed user, uint256 amount);
    event Withdrawn(address indexed user, uint256 amount);

    event TransferAutoApproved(
        address indexed sender,
        address indexed recipient,
        uint256 amount,
        uint256 riskScore
    );

    event TransferPending(
        uint256 indexed txId,
        address indexed sender,
        address indexed recipient,
        uint256 amount,
        uint256 riskScore,
        bool requiresMultiSig
    );

    event TransferExecuted(uint256 indexed txId);
    event TransferCancelled(uint256 indexed txId);
    event GuardianApproval(uint256 indexed txId, address indexed guardian);

    event GuardianAdded(address indexed guardian);
    event GuardianRemoved(address indexed guardian);

    event TrustedRecipientSet(
        address indexed user,
        address indexed recipient,
        bool trusted
    );

    event ModelUpdated(uint256 timestamp);
    event ThresholdsUpdated(uint256 lowThreshold, uint256 highThreshold);
    event TimelockDurationUpdated(uint256 newDuration);
    event MinApprovalsUpdated(uint256 newMinApprovals);

    event RiskAssessed(
        address indexed sender,
        address indexed recipient,
        uint256 amount,
        uint256 riskScore,
        int256[4] features
    );

    // ================================================================
    //                        CONSTRUCTOR
    // ================================================================

    /**
     * @param _usdc Address of the USDC (or any ERC-20) token contract
     */
    constructor(address _usdc) Ownable(msg.sender) {
        if (_usdc == address(0)) revert ZeroAddress();
        usdc = IERC20(_usdc);
        _initializeDefaultModel();
    }

    // ================================================================
    //                      USER FUNCTIONS
    // ================================================================

    /// @notice Register the caller so the guardian can build a behavioural profile
    function register() external {
        if (profiles[msg.sender].isRegistered) revert AlreadyRegistered();
        profiles[msg.sender].isRegistered = true;
        profiles[msg.sender].recentWindowStart = block.timestamp;
        emit Registered(msg.sender);
    }

    /**
     * @notice Deposit USDC into the guardian vault
     * @param amount Amount in USDC base units (6 decimals)
     */
    function deposit(uint256 amount) external nonReentrant whenNotPaused {
        if (amount == 0) revert ZeroAmount();
        if (!profiles[msg.sender].isRegistered) revert NotRegistered();

        balances[msg.sender] += amount;
        usdc.safeTransferFrom(msg.sender, address(this), amount);

        emit Deposited(msg.sender, amount);
    }

    /**
     * @notice Withdraw USDC from the vault (bypasses neural assessment)
     * @param amount Amount to withdraw
     */
    function withdraw(uint256 amount) external nonReentrant whenNotPaused {
        if (amount == 0) revert ZeroAmount();
        if (balances[msg.sender] < amount) {
            revert InsufficientBalance(balances[msg.sender], amount);
        }

        balances[msg.sender] -= amount;
        usdc.safeTransfer(msg.sender, amount);

        emit Withdrawn(msg.sender, amount);
    }

    /**
     * @notice Transfer USDC through the neural risk-assessment pipeline
     * @param recipient Destination address
     * @param amount    USDC amount (6 decimals)
     * @return txId     Pending-transaction ID (0 when auto-approved)
     * @return riskScore Computed risk score [0, 100]
     */
    function transfer(
        address recipient,
        uint256 amount
    ) external nonReentrant whenNotPaused returns (uint256 txId, uint256 riskScore) {
        if (amount == 0) revert ZeroAmount();
        if (recipient == address(0)) revert ZeroAddress();
        if (recipient == msg.sender) revert SelfTransferNotAllowed();
        if (!profiles[msg.sender].isRegistered) revert NotRegistered();
        if (balances[msg.sender] < amount) {
            revert InsufficientBalance(balances[msg.sender], amount);
        }

        // --- Neural risk assessment ---
        int256[4] memory features = _extractFeatures(msg.sender, recipient, amount);
        riskScore = _forwardPass(features);

        emit RiskAssessed(msg.sender, recipient, amount, riskScore, features);

        // Update behavioural profile *after* feature extraction
        _updateProfile(msg.sender, amount);

        if (riskScore < lowThreshold) {
            // Low risk -> instant transfer
            balances[msg.sender] -= amount;
            balances[recipient] += amount;

            emit TransferAutoApproved(msg.sender, recipient, amount, riskScore);
            return (0, riskScore);
        }

        // Medium or high risk -> lock funds and create pending tx
        balances[msg.sender] -= amount;

        txId = pendingTxCount++;
        pendingTxs[txId] = PendingTx({
            sender: msg.sender,
            recipient: recipient,
            amount: amount,
            riskScore: riskScore,
            createdAt: block.timestamp,
            approvalCount: 0,
            executed: false,
            cancelled: false
        });

        bool requiresMultiSig = riskScore >= highThreshold;

        emit TransferPending(
            txId, msg.sender, recipient, amount, riskScore, requiresMultiSig
        );

        return (txId, riskScore);
    }

    /**
     * @notice Execute a pending transfer after its security gate clears
     * @param txId Pending-transaction identifier
     */
    function executePending(uint256 txId) external nonReentrant {
        PendingTx storage ptx = pendingTxs[txId];
        if (ptx.sender == address(0)) revert TxDoesNotExist();
        if (ptx.executed) revert TxAlreadyExecuted();
        if (ptx.cancelled) revert TxAlreadyCancelled();

        if (ptx.riskScore >= highThreshold) {
            // High risk -> verify guardian quorum
            if (ptx.approvalCount < minApprovals) {
                revert InsufficientApprovals(ptx.approvalCount, minApprovals);
            }
        } else {
            // Medium risk -> verify timelock
            uint256 readyAt = ptx.createdAt + timelockDuration;
            if (block.timestamp < readyAt) {
                revert TimelockNotExpired(readyAt);
            }
        }

        ptx.executed = true;
        balances[ptx.recipient] += ptx.amount;

        emit TransferExecuted(txId);
    }

    /**
     * @notice Cancel a pending transfer and refund the sender
     * @param txId Pending-transaction identifier
     */
    function cancelPending(uint256 txId) external nonReentrant {
        PendingTx storage ptx = pendingTxs[txId];
        if (ptx.sender != msg.sender) revert NotPendingTxSender();
        if (ptx.executed) revert TxAlreadyExecuted();
        if (ptx.cancelled) revert TxAlreadyCancelled();

        ptx.cancelled = true;
        balances[msg.sender] += ptx.amount;

        emit TransferCancelled(txId);
    }

    /**
     * @notice Guardian approves a high-risk pending transfer
     * @param txId Pending-transaction identifier
     */
    function approvePending(uint256 txId) external {
        if (!guardians[msg.sender]) revert NotGuardian();

        PendingTx storage ptx = pendingTxs[txId];
        if (ptx.sender == address(0)) revert TxDoesNotExist();
        if (ptx.executed) revert TxAlreadyExecuted();
        if (ptx.cancelled) revert TxAlreadyCancelled();
        if (hasApproved[txId][msg.sender]) revert AlreadyApproved();

        hasApproved[txId][msg.sender] = true;
        ptx.approvalCount++;

        emit GuardianApproval(txId, msg.sender);
    }

    /**
     * @notice Mark a recipient as trusted (or revoke trust)
     * @param recipient Target address
     * @param trusted   true = trust, false = revoke
     */
    function setTrustedRecipient(address recipient, bool trusted) external {
        if (recipient == address(0)) revert ZeroAddress();
        trustedRecipients[msg.sender][recipient] = trusted;
        emit TrustedRecipientSet(msg.sender, recipient, trusted);
    }

    // ================================================================
    //                   NEURAL NETWORK ENGINE
    // ================================================================

    /**
     * @dev Run the 4-8-1 feedforward network and return a risk score [0, 100].
     * @param features Fixed-point input vector
     */
    function _forwardPass(
        int256[4] memory features
    ) internal view returns (uint256) {
        // ---- Hidden layer (ReLU) ----
        int256[8] memory hidden;
        for (uint256 i = 0; i < HIDDEN_SIZE; i++) {
            int256 sum = biasesHidden[i];
            uint256 base = i * INPUT_SIZE;
            for (uint256 j = 0; j < INPUT_SIZE; j++) {
                sum += _mulFP(features[j], weightsHidden[base + j]);
            }
            hidden[i] = sum > 0 ? sum : int256(0); // ReLU
        }

        // ---- Output layer (sigmoid) ----
        int256 output = biasOutput;
        for (uint256 i = 0; i < HIDDEN_SIZE; i++) {
            output += _mulFP(hidden[i], weightsOutput[i]);
        }

        // Map sigmoid [0, SCALE] -> [0, 100]
        int256 sig = _sigmoid(output);
        uint256 risk = uint256((sig * int256(MAX_RISK)) / SCALE);
        return risk > MAX_RISK ? MAX_RISK : risk;
    }

    /**
     * @notice Preview the risk score for a hypothetical transfer (view-only)
     * @param sender    Sender address
     * @param recipient Recipient address
     * @param amount    Transfer amount
     * @return riskScore  Computed risk [0, 100]
     * @return features   Extracted feature vector
     */
    function assessRisk(
        address sender,
        address recipient,
        uint256 amount
    ) external view returns (uint256 riskScore, int256[4] memory features) {
        features = _extractFeatures(sender, recipient, amount);
        riskScore = _forwardPass(features);
    }

    /**
     * @notice Run the NN on arbitrary features (useful for off-chain testing)
     * @param features 4-element fixed-point vector
     * @return riskScore Computed risk [0, 100]
     */
    function computeRisk(
        int256[4] memory features
    ) external view returns (uint256 riskScore) {
        return _forwardPass(features);
    }

    // ================================================================
    //                    FEATURE EXTRACTION
    // ================================================================

    /**
     * @dev Build the 4-feature input vector from on-chain state.
     */
    function _extractFeatures(
        address sender,
        address recipient,
        uint256 amount
    ) internal view returns (int256[4] memory features) {
        UserProfile storage profile = profiles[sender];

        // Feature 0: amountRatio = currentAmount / historicalAverage
        if (profile.totalTransactions > 0 && profile.totalVolume > 0) {
            uint256 avg = profile.totalVolume / profile.totalTransactions;
            int256 ratio = int256((amount * uint256(SCALE)) / avg);
            features[0] = ratio > MAX_AMOUNT_RATIO ? MAX_AMOUNT_RATIO : ratio;
        } else {
            features[0] = 2 * SCALE; // conservative default for new users
        }

        // Feature 1: txFrequency = recentTxCount / 5 (normalised)
        uint256 recentCount = _getRecentTxCount(sender);
        int256 freq = int256((recentCount * uint256(SCALE)) / 5);
        features[1] = freq > MAX_FREQUENCY ? MAX_FREQUENCY : freq;

        // Feature 2: recipientTrust (binary)
        features[2] = trustedRecipients[sender][recipient]
            ? SCALE
            : int256(0);

        // Feature 3: recency = 1 - timeSinceLastTx / 1 hour  (clamped [0, 1])
        if (profile.lastTransactionTime > 0) {
            uint256 delta = block.timestamp - profile.lastTransactionTime;
            if (delta >= 1 hours) {
                features[3] = 0;
            } else {
                features[3] = SCALE - int256((delta * uint256(SCALE)) / 1 hours);
            }
        } else {
            features[3] = 0; // first tx is not "recent"
        }
    }

    /// @dev Return the number of transactions in the current 1-hour rolling window.
    function _getRecentTxCount(address user) internal view returns (uint256) {
        UserProfile storage p = profiles[user];
        if (block.timestamp - p.recentWindowStart > 1 hours) return 0;
        return p.recentTxCount;
    }

    /// @dev Update the sender's behavioural profile after a transfer.
    function _updateProfile(address user, uint256 amount) internal {
        UserProfile storage p = profiles[user];

        p.totalTransactions++;
        p.totalVolume += amount;

        // Rolling 1-hour window
        if (block.timestamp - p.recentWindowStart > 1 hours) {
            p.recentTxCount = 1;
            p.recentWindowStart = block.timestamp;
        } else {
            p.recentTxCount++;
        }

        p.lastTransactionTime = block.timestamp;
    }

    // ================================================================
    //                     FIXED-POINT MATH
    // ================================================================

    /// @dev Multiply two SCALE-denominated fixed-point numbers.
    function _mulFP(int256 a, int256 b) internal pure returns (int256) {
        return (a * b) / SCALE;
    }

    /**
     * @dev Fast sigmoid: sigma(x) ~ 0.5 + 0.5 * x / (1 + |x|)
     *      Returns a value in [0, SCALE] representing the range [0.0, 1.0].
     *      The denominator (SCALE + |x|) is always > 0, so division is safe.
     */
    function _sigmoid(int256 x) internal pure returns (int256) {
        int256 absX = x >= 0 ? x : -x;
        return (SCALE / 2) + (x * (SCALE / 2)) / (SCALE + absX);
    }

    // ================================================================
    //                     ADMIN FUNCTIONS
    // ================================================================

    /**
     * @notice Replace all neural-network weights (owner only, rate-limited)
     */
    function updateModel(
        int256[32] calldata _wH,
        int256[8]  calldata _bH,
        int256[8]  calldata _wO,
        int256 _bO
    ) external onlyOwner {
        if (lastModelUpdate != 0 && block.timestamp < lastModelUpdate + MODEL_UPDATE_COOLDOWN) {
            revert ModelUpdateTooFrequent(lastModelUpdate + MODEL_UPDATE_COOLDOWN);
        }

        for (uint256 i = 0; i < 32; i++) weightsHidden[i] = _wH[i];
        for (uint256 i = 0; i < 8; i++) {
            biasesHidden[i] = _bH[i];
            weightsOutput[i] = _wO[i];
        }
        biasOutput = _bO;
        lastModelUpdate = block.timestamp;

        emit ModelUpdated(block.timestamp);
    }

    /// @notice Add a guardian for multi-sig approvals
    function addGuardian(address guardian) external onlyOwner {
        if (guardian == address(0)) revert ZeroAddress();
        if (guardians[guardian]) revert AlreadyGuardian();
        guardians[guardian] = true;
        guardianCount++;
        emit GuardianAdded(guardian);
    }

    /// @notice Remove a guardian
    function removeGuardian(address guardian) external onlyOwner {
        if (!guardians[guardian]) revert NotGuardian();
        if (guardianCount <= minApprovals) revert CannotRemoveLastGuardian();
        guardians[guardian] = false;
        guardianCount--;
        emit GuardianRemoved(guardian);
    }

    /// @notice Update the low / high risk thresholds
    function setThresholds(uint256 _low, uint256 _high) external onlyOwner {
        if (_low >= _high || _high > MAX_RISK) revert InvalidThresholds();
        lowThreshold = _low;
        highThreshold = _high;
        emit ThresholdsUpdated(_low, _high);
    }

    /// @notice Update the timelock duration for medium-risk transfers
    function setTimelockDuration(uint256 _duration) external onlyOwner {
        timelockDuration = _duration;
        emit TimelockDurationUpdated(_duration);
    }

    /// @notice Update the minimum guardian approvals for high-risk transfers
    function setMinApprovals(uint256 _min) external onlyOwner {
        if (_min == 0) revert ZeroAmount();
        minApprovals = _min;
        emit MinApprovalsUpdated(_min);
    }

    /// @notice Pause all deposits and transfers
    function pause() external onlyOwner { _pause(); }

    /// @notice Resume operations
    function unpause() external onlyOwner { _unpause(); }

    // ================================================================
    //                    VIEW HELPERS
    // ================================================================

    /// @notice Return the full set of model weights for transparency
    function getModelWeights()
        external
        view
        returns (
            int256[32] memory wH,
            int256[8]  memory bH,
            int256[8]  memory wO,
            int256 bO
        )
    {
        for (uint256 i = 0; i < 32; i++) wH[i] = weightsHidden[i];
        for (uint256 i = 0; i < 8; i++) {
            bH[i] = biasesHidden[i];
            wO[i] = weightsOutput[i];
        }
        bO = biasOutput;
    }

    /// @notice Return a user's full profile
    function getProfile(address user) external view returns (UserProfile memory) {
        return profiles[user];
    }

    /// @notice Return a pending transaction
    function getPendingTx(uint256 txId) external view returns (PendingTx memory) {
        return pendingTxs[txId];
    }

    // ================================================================
    //                  DEFAULT MODEL WEIGHTS
    // ================================================================

    /**
     * @dev Initialise the neural network with hand-tuned weights.
     *
     *  Hidden neurons and their roles:
     *    0 - Large-amount detector       (fires when amountRatio > 1.0)
     *    1 - High-frequency detector     (fires when txFrequency > 0.5)
     *    2 - Unknown-recipient alarm     (fires when recipient is NOT trusted)
     *    3 - Recent-transaction alarm    (fires when recency > 0.5)
     *    4 - Amount + frequency combo    (cross-feature interaction)
     *    5 - Untrusted + recent combo    (cross-feature interaction)
     *    6 - Trust safety net            (fires when recipient IS trusted, reduces risk)
     *    7 - General suspicion baseline
     *
     *  Output weights are tuned so that:
     *    - Normal trusted tx           -> risk ~16-21 (auto-approve)
     *    - First tx to untrusted       -> risk ~56-66 (timelock)
     *    - Large amount to untrusted   -> risk ~85+   (multi-sig)
     */
    function _initializeDefaultModel() internal {
        // --- Neuron 0: Large-amount detector ---
        weightsHidden[0]  = 2 * SCALE;     // amountRatio
        weightsHidden[1]  = 0;             // txFrequency
        weightsHidden[2]  = 0;             // recipientTrust
        weightsHidden[3]  = 0;             // recency
        biasesHidden[0]   = -2 * SCALE;    // fires when amountRatio > 1.0

        // --- Neuron 1: High-frequency detector ---
        weightsHidden[4]  = 0;
        weightsHidden[5]  = 3 * SCALE;
        weightsHidden[6]  = 0;
        weightsHidden[7]  = 0;
        biasesHidden[1]   = -15e17;        // -1.5, fires when freq > 0.5

        // --- Neuron 2: Unknown-recipient alarm ---
        weightsHidden[8]  = 0;
        weightsHidden[9]  = 0;
        weightsHidden[10] = -2 * SCALE;
        weightsHidden[11] = 0;
        biasesHidden[2]   = 2 * SCALE;     // fires when trust = 0

        // --- Neuron 3: Recent-transaction alarm ---
        weightsHidden[12] = 0;
        weightsHidden[13] = 0;
        weightsHidden[14] = 0;
        weightsHidden[15] = 2 * SCALE;
        biasesHidden[3]   = -1 * SCALE;    // fires when recency > 0.5

        // --- Neuron 4: Amount + frequency combo ---
        weightsHidden[16] = 15e17;         // 1.5
        weightsHidden[17] = 15e17;         // 1.5
        weightsHidden[18] = 0;
        weightsHidden[19] = 0;
        biasesHidden[4]   = -3 * SCALE;

        // --- Neuron 5: Untrusted + recent combo ---
        weightsHidden[20] = 0;
        weightsHidden[21] = 0;
        weightsHidden[22] = -15e17;        // -1.5
        weightsHidden[23] = 15e17;         //  1.5
        biasesHidden[5]   = -5e17;         // -0.5

        // --- Neuron 6: Trust safety net (REDUCES risk) ---
        weightsHidden[24] = 0;
        weightsHidden[25] = 0;
        weightsHidden[26] = 3 * SCALE;
        weightsHidden[27] = 0;
        biasesHidden[6]   = -1 * SCALE;    // fires when trust > 0.33

        // --- Neuron 7: General suspicion baseline ---
        weightsHidden[28] = 5e17;          //  0.5
        weightsHidden[29] = 3e17;          //  0.3
        weightsHidden[30] = -5e17;         // -0.5
        weightsHidden[31] = 3e17;          //  0.3
        biasesHidden[7]   = -3e17;         // -0.3

        // --- Output layer ---
        weightsOutput[0] = 15e16;          //  0.15  (large amount)
        weightsOutput[1] = 12e16;          //  0.12  (high frequency)
        weightsOutput[2] = 20e16;          //  0.20  (unknown recipient)
        weightsOutput[3] = 12e16;          //  0.12  (recent tx)
        weightsOutput[4] = 20e16;          //  0.20  (amount+freq combo)
        weightsOutput[5] = 25e16;          //  0.25  (untrusted+recent)
        weightsOutput[6] = -70e16;         // -0.70  (trust safety net)
        weightsOutput[7] = 12e16;          //  0.12  (general suspicion)
        biasOutput        = -3e17;         // -0.30
    }
}
