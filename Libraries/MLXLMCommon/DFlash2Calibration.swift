// Copyright © 2026 Apple Inc.

// What a DFlash2 round costs on this host, and whether this GPU's verify pass
// and commit agree with plain decoding. Measured once per drafter attach.

import Foundation
import MLX

/// Why a host's verify pass or commit disagrees with its plain steps.
public enum DFlash2SelfCheckFailure: Sendable, Equatable, CustomStringConvertible {
    /// Verify row `row` holds a NaN or an infinity.
    case nonFiniteVerifyRow(row: Int)
    /// Verify row `row` picks another token than the plain step at its
    /// position, whose top two logits are `plainGap` apart.
    case verifyDisagrees(row: Int, verifyToken: Int, plainToken: Int, plainGap: Float)
    /// After committing `committed` rows, the next step's logits are not finite.
    case nonFiniteAfterCommit(committed: Int)
    /// After committing `committed` rows, the next step picks another token
    /// than the plain step at that position.
    case commitDisagrees(committed: Int, token: Int, plainToken: Int, plainGap: Float)

    public var description: String {
        switch self {
        case .nonFiniteVerifyRow(let row):
            "verify row \(row) is not finite"
        case .verifyDisagrees(let row, let verify, let plain, let gap):
            "verify row \(row) picks \(verify) where the plain step picks \(plain) (top-2 gap \(gap))"
        case .nonFiniteAfterCommit(let committed):
            "the step after committing \(committed) rows is not finite"
        case .commitDisagrees(let committed, let token, let plain, let gap):
            "after committing \(committed) rows the step picks \(token) where the plain step picks \(plain) (top-2 gap \(gap))"
        }
    }
}

/// One host's DFlash2 round cost and verify self-check, from
/// ``dflash2Calibrate(target:drafter:blockSize:parameters:)``.
public struct DFlash2Calibration: Sendable, Equatable {
    public enum Outcome: Sendable, Equatable {
        /// The timing is steady and the self-check agrees: speculate with
        /// ``fallback``.
        case passed
        /// Both measurements were unsteady: plain or round intervals spread,
        /// or the plain steps after the rounds ran at another speed than
        /// before them. No self-check ran.
        case noisy
        /// A round costs at least `blockSize` plain steps; no self-check ran.
        case cannotPay
        /// This GPU's verify pass or commit disagrees with its plain steps.
        case selfCheckFailed(DFlash2SelfCheckFailure)
    }

    /// Median round interval over the mean of the two plain medians.
    public var roundCost: Double
    /// Mean of the medians of the plain intervals before and after the rounds.
    public var plainStepSeconds: Double
    /// Median sync-to-sync round interval.
    public var roundSeconds: Double
    /// Larger of the two plain phases' median absolute deviation over median.
    public var plainSpread: Double
    /// `|after − before| / mean` of the two plain medians: a level shift
    /// across the measurement.
    public var plainDrift: Double
    /// Median absolute deviation over median of the round intervals.
    public var roundSpread: Double
    public var blockSize: Int
    /// 1, or 2 when the first measurement was noisy.
    public var measurements: Int
    public var outcome: Outcome

    /// ``plainSpread`` above which a measurement is repeated once, then declined.
    public static let noiseThreshold = 0.10
    /// ``plainDrift`` above which a measurement is repeated once, then declined.
    public static let driftThreshold = 0.05
    /// ``roundSpread`` above which a measurement is repeated once, then declined.
    public static let roundSpreadThreshold = 0.10
    /// Untimed plain steps, then untimed rounds, before the timing: they
    /// compile the traces and cover the first decode's ramp, which on the
    /// M5 Ultra ran six steps from 26 to 21 ms after a cold load.
    public static let warmupPlainSteps = 8
    public static let warmupRounds = 4
    /// Plain steps per timed phase.
    public static let plainIntervals = 8
    /// Rounds timed between the plain phases.
    public static let roundIntervals = 8
    /// A plain step whose top two logits are closer than this may pick
    /// another token under another summation order, as the verify pass's.
    public static let nearTieGap: Float = 0.5

    /// The fallback a turn at ``blockSize`` uses on this host.
    public var fallback: DFlash2Fallback {
        DFlash2Fallback(roundCost: roundCost, blockSize: blockSize)
    }

    /// Whether one measurement is too unsteady to trust; a NaN, as from a
    /// zero median, is.
    static func isNoisy(_ summary: Summary) -> Bool {
        !(summary.plainSpread <= noiseThreshold && summary.plainDrift <= driftThreshold
            && summary.roundSpread <= roundSpreadThreshold)
    }

    typealias Summary = (
        roundCost: Double, plainStepSeconds: Double, roundSeconds: Double, plainSpread: Double,
        plainDrift: Double, roundSpread: Double
    )

    /// Medians of one measurement. `before` and `after` bracket the rounds, so
    /// a drift linear in time cancels from their mean.
    static func summarize(before: [Double], rounds: [Double], after: [Double]) -> Summary {
        let plain = (median(before) + median(after)) / 2
        let round = median(rounds)
        let spread = Swift.max(relativeDeviation(before), relativeDeviation(after))
        let drift = abs(median(after) - median(before)) / plain
        return (round / plain, plain, round, spread, drift, relativeDeviation(rounds))
    }

    static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count % 2 == 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    }

    static func relativeDeviation(_ values: [Double]) -> Double {
        let center = median(values)
        return median(values.map { abs($0 - center) }) / center
    }
}

/// Measure what a DFlash2 round costs on this host and check this GPU's
/// verify pass and commit against plain decoding.
///
/// Runs over a fixed 33-token prompt on its own cache, which it drops before
/// returning; the target keeps only the compiled decode and verify traces the
/// first turn would build. Untimed plain steps and rounds first bring a cold
/// process to speed; timing then runs plain steps, rounds, plain steps. Plain
/// steps are ``TokenIterator``'s, as a plain turn decodes, sampled with
/// `parameters`; rounds are ``DFlash2TokenIterator``'s at `blockSize` with
/// fallback off, timed sync to sync. An unsteady measurement (spread plain or
/// round intervals, or plain medians that shift across the rounds) is
/// measured once more, then declined as ``DFlash2Calibration/Outcome/noisy``.
///
/// The self-check takes `blockSize` plain greedy steps from one state, runs one
/// verify pass over the tokens they fed, on the route a round takes, and
/// requires every row finite with the plain step's argmax unless that step's
/// top two nearly tie. It then commits each count from the whole block down
/// to one row, as a round commits, and requires each next step's argmax to
/// match the plain step at that position.
///
/// - Parameters:
///   - parameters: the sampling turns use; `maxTokens` and the prefill
///     callbacks are ignored.
/// - Throws: ``DFlash2SpeculationError`` when the pair cannot speculate at all.
public func dflash2Calibrate(
    target: any LanguageModel, drafter: any DFlash2DrafterModel, blockSize: Int,
    parameters: GenerateParameters
) throws -> DFlash2Calibration {
    try dflash2Calibrate(
        target: target, drafter: drafter, blockSize: blockSize, parameters: parameters,
        clock: { ProcessInfo.processInfo.systemUptime })
}

/// ``dflash2Calibrate(target:drafter:blockSize:parameters:)`` with the clock,
/// tie gap and self-check faults injectable.
func dflash2Calibrate(
    target: any LanguageModel, drafter: any DFlash2DrafterModel, blockSize: Int,
    parameters: GenerateParameters, clock: () -> Double,
    nearTieGap: Float = DFlash2Calibration.nearTieGap, faults: DFlash2SelfCheckFaults = .init()
) throws -> DFlash2Calibration {
    guard let dflash2Target = target as? any DFlash2TargetModel else {
        throw DFlash2SpeculationError.unsupportedTarget
    }
    var parameters = parameters
    parameters.maxTokens = nil
    parameters.prefill.progress = nil
    parameters.prefill.prefix = nil
    let prefix = try DFlash2CalibrationPrefix(
        target: target, vocabularySize: dflash2Target.dflash2Geometry.vocabularySize,
        parameters: parameters)

    _ = try prefix.plainIntervals(
        target: target, parameters: parameters, skipping: DFlash2Calibration.warmupPlainSteps,
        count: 0, clock: clock)
    _ = try prefix.roundIntervals(
        target: target, drafter: drafter, blockSize: blockSize, parameters: parameters,
        skipping: DFlash2Calibration.warmupRounds, count: 0, clock: clock)

    var measurements = 0
    var summary = DFlash2Calibration.summarize(before: [1], rounds: [1], after: [1])
    repeat {
        measurements += 1
        let plainBefore = try prefix.plainIntervals(
            target: target, parameters: parameters, clock: clock)
        let rounds = try prefix.roundIntervals(
            target: target, drafter: drafter, blockSize: blockSize, parameters: parameters,
            clock: clock)
        let plainAfter = try prefix.plainIntervals(
            target: target, parameters: parameters, clock: clock)
        summary = DFlash2Calibration.summarize(
            before: plainBefore, rounds: rounds, after: plainAfter)
    } while DFlash2Calibration.isNoisy(summary) && measurements < 2

    var calibration = DFlash2Calibration(
        roundCost: summary.roundCost, plainStepSeconds: summary.plainStepSeconds,
        roundSeconds: summary.roundSeconds, plainSpread: summary.plainSpread,
        plainDrift: summary.plainDrift, roundSpread: summary.roundSpread, blockSize: blockSize,
        measurements: measurements, outcome: .passed)
    if DFlash2Calibration.isNoisy(summary) {
        calibration.outcome = .noisy
    } else if !(summary.roundCost < Double(blockSize)) {
        calibration.outcome = .cannotPay
    } else if let failure = try dflash2SelfCheck(
        target: dflash2Target, prefix: prefix, blockSize: blockSize,
        captureLayers: drafter.targetLayerIds, nearTieGap: nearTieGap, faults: faults)
    {
        calibration.outcome = .selfCheckFailed(failure)
    }
    return calibration
}

/// Defects a test plants in the self-check.
struct DFlash2SelfCheckFaults {
    /// Rewrites the verify logits before they are checked.
    var verifyLogits: ((MLXArray) -> MLXArray)?
    /// Added to the count the self-check commits.
    var commitSkew = 0
}

/// The calibration prompt but its last token, prefilled once; every phase
/// decodes from a copy.
struct DFlash2CalibrationPrefix {
    /// `1, 2, …, 10, 11` in byte-level BPE alphabet ids, which the Qwen
    /// tokenizers share and which is how they split digits, commas and the
    /// space before a digit.
    ///
    /// Its greedy continuation counts on, each token confident and unlike its
    /// neighbours, so a verify row or commit off by one position picks another
    /// token. Over a pseudorandom prompt the 27B continues `" 1000…"`: rows
    /// that near-tie or repeat one token, which no shift changes.
    static let prompt: [Int] = (1 ... 11).map(String.init).joined(separator: ", ").unicodeScalars
        .map { $0 == " " ? 220 : Int($0.value) - 33 }

    let prompt = Self.prompt
    let cache: [KVCache]
    let state: LMOutput.State?

    init(target: any LanguageModel, vocabularySize: Int, parameters: GenerateParameters) throws {
        guard Self.prompt.allSatisfy({ $0 < vocabularySize }) else {
            throw DFlash2SpeculationError.unsupportedTarget
        }
        cache = try target.newCache(parameters: parameters)
        guard
            case .logits(let output) = try target.prepare(
                LMInput(tokens: MLXArray(prompt.dropLast().map(Int32.init))), cache: cache,
                state: nil, prefill: .init())
        else { throw DFlash2SpeculationError.unsupportedTarget }
        eval(cache.flatMap { $0.innerState() })
        state = output.state
    }

    var position: Int { prompt.count - 1 }

    func copiedCache() -> [KVCache] { cache.map { $0.copy() } }

    /// `count` intervals between ``TokenIterator`` returns, after `skipping`
    /// untimed steps; the clock is not read when `count` is 0.
    func plainIntervals(
        target: any LanguageModel, parameters: GenerateParameters, skipping: Int = 1,
        count: Int = DFlash2Calibration.plainIntervals, clock: () -> Double
    ) throws -> [Double] {
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([Int32(prompt[position])])), model: target,
            cache: copiedCache(), state: state, parameters: parameters)
        for _ in 0 ... skipping {
            _ = iterator.next()
        }
        guard count > 0 else { return [] }
        var stamps = [clock()]
        for _ in 0 ..< count {
            _ = iterator.next()
            stamps.append(clock())
        }
        return zip(stamps.dropFirst(), stamps).map { $0 - $1 }
    }

    /// `count` intervals between round syncs, after `skipping` untimed rounds;
    /// the clock is not read when `count` is 0.
    func roundIntervals(
        target: any LanguageModel, drafter: any DFlash2DrafterModel, blockSize: Int,
        parameters: GenerateParameters, skipping: Int = 1,
        count: Int = DFlash2Calibration.roundIntervals, clock: () -> Double
    ) throws -> [Double] {
        var iterator = try DFlash2TokenIterator(
            input: LMInput(tokens: MLXArray(prompt.map(Int32.init))), mainModel: target,
            drafter: drafter, mainCache: copiedCache(), prefilledPrefixTokens: position,
            state: state, parameters: parameters, blockSize: blockSize, fallback: .disabled)
        func syncs() -> Int { iterator.speculativeDecodingTelemetry?.roundCount ?? 0 }
        while syncs() < skipping, iterator.next() != nil {}
        guard count > 0 else { return [] }
        var stamps: [Double] = []
        var rounds = syncs()
        while stamps.count < count + 1, iterator.next() != nil {
            if syncs() > rounds {
                rounds = syncs()
                stamps.append(clock())
            }
        }
        return zip(stamps.dropFirst(), stamps).map { $0 - $1 }
    }
}

/// `nil` when this GPU's verify pass and partial commit agree with its plain
/// steps from one state; the first disagreement otherwise.
///
/// The verify pass is a round's: it captures `captureLayers`, the drafter's, so
/// it runs the trace turns run, and it sees `blockSize − 1` masked scratch
/// columns past the block, as a round positioned behind its upper bound does.
func dflash2SelfCheck(
    target: any DFlash2TargetModel, prefix: DFlash2CalibrationPrefix, blockSize: Int,
    captureLayers: [Int], nearTieGap: Float, faults: DFlash2SelfCheckFaults = .init()
) throws -> DFlash2SelfCheckFailure? {
    func step(_ token: Int, _ cache: [KVCache], _ state: LMOutput.State?) -> (
        logits: MLXArray, state: LMOutput.State?
    ) {
        let output = target(
            LMInput.Text(tokens: MLXArray([Int32(token)]).reshaped(1, 1)), cache: cache,
            state: state)
        return (output.logits[0, -1].asType(.float32), output.state)
    }
    func topTwo(_ row: MLXArray) -> (token: Int, gap: Float) {
        let sorted = MLX.sorted(row).asArray(Float.self)
        return (argMax(row).item(Int.self), sorted[sorted.count - 1] - sorted[sorted.count - 2])
    }
    func finite(_ rows: MLXArray) -> [Bool] {
        (abs(rows) .< Float.infinity).all(axis: -1).asArray(Bool.self)
    }

    // Row `i` of the plain chain is the step after feeding `block[i]`; one
    // step past the block gives the step after a full commit.
    let plainCache = prefix.copiedCache()
    var plainState = prefix.state
    var block = [prefix.prompt[prefix.position]]
    var plain: [(token: Int, gap: Float)] = []
    for i in 0 ... blockSize {
        let output = step(block[i], plainCache, plainState)
        plainState = output.state
        plain.append(topTwo(output.logits))
        block.append(plain[i].token)
    }

    let cache = prefix.copiedCache()
    try target.dflash2Prepare(cache: cache, state: prefix.state)
    let result = target.dflash2Verify(
        DFlash2VerifyRequest(
            tokens: MLXArray(block[..<blockSize].map(Int32.init)).expandedDimensions(axis: 0),
            position: MLXArray([Int32(prefix.position)]),
            positionUpperBound: prefix.position + blockSize - 1, captureLayers: captureLayers),
        cache: cache)
    var logits = result.logits[0].asType(.float32)
    if let corrupt = faults.verifyLogits { logits = corrupt(logits) }
    for (row, isFinite) in finite(logits).enumerated() where !isFinite {
        return .nonFiniteVerifyRow(row: row)
    }
    let verifyTokens = argMax(logits, axis: -1).asType(.int32).asArray(Int32.self).map(Int.init)
    for (row, token) in verifyTokens.enumerated()
    where token != plain[row].token && plain[row].gap >= nearTieGap {
        return .verifyDisagrees(
            row: row, verifyToken: token, plainToken: plain[row].token, plainGap: plain[row].gap)
    }

    // Every count a round can commit, widest first: a step after committing
    // `committed` rows overwrites only row `committed`, which no narrower
    // commit keeps, and the replay rebuilds recurrent state from the captures.
    for committed in stride(from: blockSize, through: 1, by: -1) {
        let count = committed + faults.commitSkew
        guard (1 ... blockSize).contains(count) else { continue }
        DFlash2TokenIterator.commitRecurrentState(
            cache: cache, captures: result.recurrentCaptures, validCount: MLXArray(Int32(count)))
        for case let attention as KVCacheSimple in cache {
            attention.commitRows(count: prefix.position + count)
        }
        let next = step(block[committed], cache, prefix.state).logits
        guard finite(next.expandedDimensions(axis: 0))[0] else {
            return .nonFiniteAfterCommit(committed: committed)
        }
        let token = argMax(next).item(Int.self)
        let expected = plain[committed]
        if token != expected.token && expected.gap >= nearTieGap {
            return .commitDisagrees(
                committed: committed, token: token, plainToken: expected.token,
                plainGap: expected.gap)
        }
    }
    return nil
}

/// Times each phase of a round behind an evaluation barrier. The barriers
/// serialize host and GPU, so the phases sum to more than a pipelined round.
package final class DFlash2PhaseTimer {
    package enum Phase: String, CaseIterable, Sendable {
        case draft, verify, replay, sampling, commit
    }

    package private(set) var seconds: [Phase: [Double]] = [:]
    private var last: Double?

    package init() {}

    /// Open a round; marks before the first round opens are not recorded.
    package func start() {
        last = ProcessInfo.processInfo.systemUptime
    }

    /// Evaluate `arrays` and charge the time since the last mark to `phase`.
    package func mark(_ phase: Phase, _ arrays: [MLXArray]) {
        eval(arrays)
        guard let last else { return }
        let now = ProcessInfo.processInfo.systemUptime
        seconds[phase, default: []].append(now - last)
        self.last = now
    }

    package func reset() {
        seconds = [:]
        last = nil
    }
}
