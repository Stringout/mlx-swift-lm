// Copyright © 2026 Apple Inc.

// DFlash2 throughput bench over a local target and drafter, on prerendered
// prompts. Plain decoding is `TokenIterator`, as an ordinary generate runs it.
//
//   DFlash2Bench --target DIR --drafter DIR --tokens DIR [--files a,b]
//     [--max-tokens 512] [--blocks 5[,8]] [--temperatures 0.15] [--top-p 0.8]
//     [--seed 1] [--suffix 512] [--fallback disabled|measured]
//     [--clock-hold on|off | --clock-hold-ab] [--out results.json]
//   DFlash2Bench ... --cost-sweep [--sweep-widths 2-16] [--out sweep.json]
//   DFlash2Bench ... --files prompt-1 --length-study 30
//   DFlash2Bench ... --prefill-sweep 128,256,512,1024,2048 [--prefill-profile]
//     [--prefill-dense-rows 64] [--profile-rows 1,8,64,256,512,1024,2048]
//     [--out prefill.json]
//
// Paired mode prefills each prompt but its last `--suffix` tokens once; every
// arm decodes from a copy of that checkpoint and prefills the suffix itself,
// so the drafter sees the suffix as context, as over a production prefix
// checkpoint. Arms run in mirrored order per prompt: plain, each width, the
// widths reversed, plain. With `--clock-hold-ab`, plain off/on/on/off, then each
// width off/on/on/off, where on holds the GPU clock from the first decoded
// token to the last. Speculative arms record every round's accepted drafts.
//
// `--fallback measured` calibrates each width once at start and gates every
// speculative turn on that round cost; `disabled` never falls back.
//
// The cost sweep calibrates each width on a short fixed prompt, which gives the
// round cost sync to sync, then times one run's rounds phase by phase behind
// evaluation barriers. The barriers serialize host and GPU, so the phases sum
// to more than a pipelined round; their shares are what the sweep reports.
//
// The length study samples each prompt with seeds 0..<N at the first
// temperature in four arms: TokenIterator; speculative; plain steps with the
// iterator's per-position seeds; and those plain steps run as one-row verify
// passes, which take the verify pass's gated-delta and attention routes.
// `--warm-prefix P` instead decodes from copies of a checkpoint holding the
// first P prompt tokens, as Stringout's prefix cache does, in the first two arms.
//
// The prefill sweep prefills each whole prompt cold at each chunk ceiling, in
// mirrored order per prompt, and reports pooled tokens per second against the
// 512 default. `--prefill-profile` then times every quantized linear of the
// decoder layers alone on random rows, as the quantized kernel and as a dense
// matmul over the dequantized weight, so the matmuls' share of a chunk and the
// row count they run best at can be read apart from attention and the scan.
// `--prefill-dense-rows N` adds a second arm per ceiling in which every decoder
// linear but the fused gated-delta input projections multiplies a forward of N
// rows or more by its dequantized weight. `--prefill-stock-arm` adds a third
// arm that puts the model's own linears back first, so the quantized arm over
// swapped linears can be read against the model as it loads.
//
// A token file is `{label, phase, tokenIds, text}`.

import BenchmarkHelpers
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

struct Options {
    var target: URL
    var drafter: URL
    var tokens: URL
    var files: [String]?
    var maxTokens = 512
    var blocks = [5]
    var temperatures: [Float] = [0.15]
    var topP: Float = 0.8
    var seed: UInt64 = 1
    var suffix = 512
    var measuredFallback = false
    var clockHold = false
    var clockHoldAB = false
    var costSweep = false
    var calibrateOnly = false
    var coldTrace = false
    var sweepWidths = Array(2 ... 16)
    var lengthStudy: Int?
    var warmPrefix: Int?
    var prefillSweep: [Int]?
    var prefillProfile = false
    var profileRows = [1, 8, 64, 256, 512, 1024, 2048]
    var prefillDenseRows: Int?
    var prefillStockArm = false
    var out: URL?

    init(_ arguments: [String]) throws {
        var values: [String: String] = [:]
        var flags: Set<String> = []
        var index = 1
        while index < arguments.count {
            let key = arguments[index]
            guard key.hasPrefix("--") else { throw BenchError.usage("unexpected \(key)") }
            if index + 1 < arguments.count, !arguments[index + 1].hasPrefix("--") {
                values[String(key.dropFirst(2))] = arguments[index + 1]
                index += 2
            } else {
                flags.insert(String(key.dropFirst(2)))
                index += 1
            }
        }
        func url(_ key: String) throws -> URL {
            guard let path = values[key] else { throw BenchError.usage("--\(key) is required") }
            return URL(fileURLWithPath: path)
        }
        func integers(_ text: String) -> [Int] {
            text.split(separator: ",").flatMap { part -> [Int] in
                let bounds = part.split(separator: "-").compactMap { Int($0) }
                return bounds.count == 2 ? Array(bounds[0] ... bounds[1]) : bounds
            }
        }
        target = try url("target")
        drafter = try url("drafter")
        tokens = try url("tokens")
        files = values["files"].map { $0.split(separator: ",").map(String.init) }
        maxTokens = values["max-tokens"].flatMap(Int.init) ?? maxTokens
        blocks = values["blocks"].map(integers) ?? blocks
        temperatures =
            values["temperatures"].map { $0.split(separator: ",").compactMap { Float($0) } }
            ?? temperatures
        topP = values["top-p"].flatMap(Float.init) ?? topP
        seed = values["seed"].flatMap(UInt64.init) ?? seed
        suffix = values["suffix"].flatMap(Int.init) ?? suffix
        switch values["fallback"] ?? "disabled" {
        case "disabled": measuredFallback = false
        case "measured": measuredFallback = true
        case let other: throw BenchError.usage("--fallback disabled|measured, not \(other)")
        }
        switch values["clock-hold"] ?? "off" {
        case "off": clockHold = false
        case "on": clockHold = true
        case let other: throw BenchError.usage("--clock-hold on|off, not \(other)")
        }
        clockHoldAB = flags.contains("clock-hold-ab")
        costSweep = flags.contains("cost-sweep")
        calibrateOnly = flags.contains("calibrate-only")
        coldTrace = flags.contains("cold-trace")
        sweepWidths = values["sweep-widths"].map(integers) ?? sweepWidths
        lengthStudy = values["length-study"].flatMap(Int.init)
        warmPrefix = values["warm-prefix"].flatMap(Int.init)
        prefillSweep = values["prefill-sweep"].map(integers)
        prefillProfile = flags.contains("prefill-profile")
        profileRows = values["profile-rows"].map(integers) ?? profileRows
        prefillDenseRows = values["prefill-dense-rows"].flatMap { Int($0) }
        prefillStockArm = flags.contains("prefill-stock-arm")
        out = values["out"].map { URL(fileURLWithPath: $0) }
    }
}

enum BenchError: Error {
    case usage(String)
}

struct TokenFile: Decodable {
    let label: String
    let phase: String
    let tokenIds: [Int]
}

/// One decode of one prompt.
struct Arm: Codable {
    /// `plain` or `speculative`.
    var kind: String
    var block: Int?
    var clockHold: Bool
    var tokens: [Int]
    /// Suffix tokens this arm prefilled over the shared checkpoint.
    var prefillTokens: Int
    var prefillSeconds: Double
    var decodeSeconds: Double
    var peakMemory: Int
    var activeMemory: Int
    var cacheMemory: Int
    var rounds: Int?
    var proposed: Int?
    var accepted: Int?
    /// Accepted drafts of each round, in order.
    var roundAccepts: [Int]?
    var passthroughReason: String?

    var outputTokens: Int
    /// Tokens after the first, which prefill produces.
    var decodeTokens: Int
    var decodeRate: Double
    var prefillRate: Double
    var tau: Double?

    init(
        kind: String, block: Int?, clockHold: Bool, tokens: [Int], prefillTokens: Int,
        prefillSeconds: Double, decodeSeconds: Double, rounds: Int? = nil,
        proposed: Int? = nil, accepted: Int? = nil, roundAccepts: [Int]? = nil,
        passthroughReason: String? = nil
    ) {
        self.kind = kind
        self.block = block
        self.clockHold = clockHold
        self.tokens = tokens
        self.prefillTokens = prefillTokens
        self.prefillSeconds = prefillSeconds
        self.decodeSeconds = decodeSeconds
        self.peakMemory = Memory.peakMemory
        self.activeMemory = Memory.activeMemory
        self.cacheMemory = Memory.cacheMemory
        self.rounds = rounds
        self.proposed = proposed
        self.accepted = accepted
        self.roundAccepts = roundAccepts
        self.passthroughReason = passthroughReason
        outputTokens = tokens.count
        decodeTokens = max(0, tokens.count - 1)
        decodeRate = Double(decodeTokens) / max(decodeSeconds, 1e-9)
        prefillRate = Double(prefillTokens) / max(prefillSeconds, 1e-9)
        if let rounds, let accepted, rounds > 0 {
            tau = Double(accepted + rounds) / Double(rounds)
        }
    }
}

struct Turn: Codable {
    var label: String
    var phase: String
    var temperature: Float
    var promptTokens: Int
    /// Prompt tokens in the shared checkpoint.
    var sharedPrefix: Int
    /// In run order.
    var arms: [Arm]
}

/// Pooled over a subset of turns: ratios of sums.
struct Summary: Codable {
    var temperature: Float
    var subset: String
    var clockHold: Bool
    var block: Int
    var turns: Int
    var plainDecodeRate: Double
    var speculativeDecodeRate: Double
    var speedup: Double
    var tau: Double
    /// Seconds per round over seconds per plain step, from the same window.
    var roundCost: Double
    /// `tau / roundCost`.
    var gain: Double
    var fellBack: Int
}

/// Clock hold on over off, pooled rates.
struct HoldComparison: Codable {
    var temperature: Float
    var subset: String
    var kind: String
    var block: Int?
    var decodeRatio: Double
    var prefillRatio: Double
}

struct Calibration: Codable {
    var block: Int
    var roundCost: Double
    var plainStepSeconds: Double
    var roundSeconds: Double
    var plainSpread: Double
    var plainDrift: Double
    var roundSpread: Double
    var measurements: Int
    var outcome: String

    init(_ calibration: DFlash2Calibration) {
        block = calibration.blockSize
        roundCost = calibration.roundCost
        plainStepSeconds = calibration.plainStepSeconds
        roundSeconds = calibration.roundSeconds
        plainSpread = calibration.plainSpread
        plainDrift = calibration.plainDrift
        roundSpread = calibration.roundSpread
        measurements = calibration.measurements
        outcome = "\(calibration.outcome)"
    }
}

struct Report: Codable {
    var target: String
    var drafter: String
    var blocks: [Int]
    var maxTokens: Int
    var topP: Float
    var suffix: Int
    var fallback: String
    var calibrations: [Calibration]
    var drafterLoadSeconds: Double
    var drafterBytes: Int
    var turns: [Turn]
    var summaries: [Summary]
    var holdComparisons: [HoldComparison]
}

func now() -> Double { ProcessInfo.processInfo.systemUptime }

func generationConfigEOS(_ directory: URL) -> Set<Int> {
    let url = directory.appending(component: "generation_config.json")
    guard let data = try? Data(contentsOf: url),
        let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return [] }
    if let ids = root["eos_token_id"] as? [Int] { return Set(ids) }
    if let id = root["eos_token_id"] as? Int { return [id] }
    return []
}

/// Drain `next` until EOS or the limit, timing the first token apart.
/// `decoding` runs once the first token is out; `observe` after every token.
func measure(
    start: Double, eos: Set<Int>, decoding: () -> Void = {}, observe: () -> Void = {},
    next: () -> Int?
) -> (tokens: [Int], prefill: Double, decode: Double) {
    var tokens: [Int] = []
    var first = start
    while let token = next() {
        if tokens.isEmpty {
            first = now()
            decoding()
        }
        observe()
        tokens.append(token)
        if eos.contains(token) { break }
    }
    let end = now()
    return (tokens, first - start, end - first)
}

func resetMemory() {
    Stream.gpu.synchronize()
    Memory.clearCache()
    Memory.peakMemory = 0
}

func medianOf(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    let middle = sorted.count / 2
    return sorted.count % 2 == 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
}

/// The prompt's first tokens, prefilled once for every arm of the prompt.
struct Checkpoint {
    let prefix: Int
    let cache: [KVCache]
    let state: LMOutput.State?

    init(model: any LanguageModel, prompt: MLXArray, prefix: Int) throws {
        self.prefix = prefix
        let cache = try model.newCache(parameters: nil)
        var state: LMOutput.State? = nil
        if prefix > 0 {
            guard
                case .logits(let output) = try model.prepare(
                    LMInput(tokens: prompt[..<prefix]), cache: cache, state: nil,
                    prefill: .init())
            else { throw BenchError.usage("prepare left tokens") }
            eval(cache.flatMap { $0.innerState() })
            state = output.state
        }
        self.cache = cache
        self.state = state
    }

    /// A copy an arm may write without reaching the checkpoint or another arm.
    func copied() -> [KVCache] { cache.map { $0.copy() } }
}

struct ArmSpec {
    var kind: String
    var block: Int?
    var clockHold: Bool
}

func run(_ options: Options) async throws {
    var configuration = ResolvedModelConfiguration(directory: options.target)
    configuration.excludedComponents = [.vision]
    let context = try await VLMModelFactory.shared._load(
        configuration: configuration, tokenizerLoader: NoOpTokenizerLoader())
    let model = context.model
    let eos = generationConfigEOS(options.target)

    if options.prefillSweep != nil || options.prefillProfile {
        try prefillStudy(options: options, model: model)
        return
    }

    let drafterStart = now()
    let drafterContext = try await DFlash2DrafterModelFactory.load(from: options.drafter)
    let drafterLoadSeconds = now() - drafterStart
    let drafter = drafterContext.model
    let drafterBytes = try drafter.memory().model
    print(
        String(
            format: "drafter: %.2f s load, %.1f MiB, eos %@", drafterLoadSeconds,
            Double(drafterBytes) / 1_048_576, eos.sorted().description))

    if let seeds = options.lengthStudy {
        try lengthStudy(
            options: options, seeds: seeds, model: model, drafter: drafter, eos: eos)
        return
    }
    let hold = options.clockHold || options.clockHoldAB ? GPUClockHold() : nil
    if (options.clockHold || options.clockHoldAB) && hold == nil {
        throw BenchError.usage("no Metal device for the clock hold")
    }
    if options.costSweep {
        try costSweep(options: options, model: model, drafter: drafter, hold: hold)
        return
    }
    if options.calibrateOnly {
        var calibrations: [Calibration] = []
        for block in options.blocks {
            let start = now()
            let calibration = try dflash2Calibrate(
                target: model, drafter: drafter, blockSize: block,
                parameters: GenerateParameters(
                    temperature: options.temperatures[0], topP: options.topP))
            print(calibrationRow(calibration, seconds: now() - start))
            calibrations.append(Calibration(calibration))
        }
        try write(calibrations, to: options.out)
        return
    }
    if options.coldTrace {
        try coldTrace(options: options, model: model, drafter: drafter)
        return
    }

    var calibrations: [Int: DFlash2Calibration] = [:]
    if options.measuredFallback {
        for block in options.blocks {
            if options.clockHold { hold?.start() }
            let calibration = try dflash2Calibrate(
                target: model, drafter: drafter, blockSize: block,
                parameters: GenerateParameters(
                    temperature: options.temperatures[0], topP: options.topP))
            hold?.stop()
            print(calibrationRow(calibration, seconds: nil))
            guard calibration.outcome == .passed else {
                throw BenchError.usage("block \(block) declined at attach; nothing to gate on")
            }
            calibrations[block] = calibration
        }
    }

    var specs: [ArmSpec] = []
    if options.clockHoldAB {
        for kind in [("plain", nil)] + options.blocks.map({ ("speculative", Optional($0)) }) {
            for on in [false, true, true, false] {
                specs.append(ArmSpec(kind: kind.0, block: kind.1, clockHold: on))
            }
        }
    } else {
        let widths = options.blocks + options.blocks.reversed()
        specs =
            [ArmSpec(kind: "plain", block: nil, clockHold: options.clockHold)]
            + widths.map { ArmSpec(kind: "speculative", block: $0, clockHold: options.clockHold) }
            + [ArmSpec(kind: "plain", block: nil, clockHold: options.clockHold)]
    }

    let names =
        try options.files.map { $0.map { "\($0).json" } }
        ?? FileManager.default.contentsOfDirectory(atPath: options.tokens.path)
        .filter { $0.hasSuffix(".json") }.sorted()

    var turns: [Turn] = []
    for name in names {
        let file = try JSONDecoder().decode(
            TokenFile.self, from: Data(contentsOf: options.tokens.appending(component: name)))
        let prompt = MLXArray(file.tokenIds.map(Int32.init))
        let checkpoint = try Checkpoint(
            model: model, prompt: prompt, prefix: max(0, file.tokenIds.count - options.suffix))
        for temperature in options.temperatures {
            var parameters = GenerateParameters(
                maxTokens: options.maxTokens, temperature: temperature, topP: options.topP)
            parameters.seed = options.seed
            var arms: [Arm] = []
            for spec in specs {
                let fallback =
                    spec.block.flatMap { calibrations[$0]?.fallback } ?? DFlash2Fallback.disabled
                let arm = try runArm(
                    spec, prompt: prompt, checkpoint: checkpoint, parameters: parameters,
                    model: model, drafter: drafter, eos: eos, fallback: fallback,
                    hold: spec.clockHold ? hold : nil)
                arms.append(arm)
                print(row(file.label, temperature, arm))
                fflush(stdout)
            }
            turns.append(
                Turn(
                    label: file.label, phase: file.phase, temperature: temperature,
                    promptTokens: file.tokenIds.count, sharedPrefix: checkpoint.prefix,
                    arms: arms))
        }
    }

    let (summaries, comparisons) = summarize(turns, options: options)
    print("")
    print(
        "T     subset   hold block n   plain tok/s  spec tok/s  speedup  tau   R     tau/R  fell back"
    )
    for summary in summaries {
        print(
            String(
                format: "%-5.2f %-8@ %-4@ %-5d %-3d %-12.1f %-11.1f %-8.2f %-5.2f %-5.2f %-6.2f %d",
                summary.temperature, summary.subset as NSString,
                (summary.clockHold ? "on" : "off") as NSString, summary.block, summary.turns,
                summary.plainDecodeRate, summary.speculativeDecodeRate, summary.speedup,
                summary.tau, summary.roundCost, summary.gain, summary.fellBack))
    }
    if !comparisons.isEmpty {
        print("")
        print("clock hold on / off, pooled")
        print("T     subset   arm          decode  prefill")
        for comparison in comparisons {
            let arm = comparison.block.map { "spec \($0)" } ?? "plain"
            print(
                String(
                    format: "%-5.2f %-8@ %-12@ %-7.4f %.4f", comparison.temperature,
                    comparison.subset as NSString, arm as NSString, comparison.decodeRatio,
                    comparison.prefillRatio))
        }
    }

    let report = Report(
        target: options.target.path, drafter: options.drafter.path, blocks: options.blocks,
        maxTokens: options.maxTokens, topP: options.topP, suffix: options.suffix,
        fallback: options.measuredFallback ? "measured" : "disabled",
        calibrations: options.blocks.compactMap { calibrations[$0] }.map(Calibration.init),
        drafterLoadSeconds: drafterLoadSeconds, drafterBytes: drafterBytes, turns: turns,
        summaries: summaries, holdComparisons: comparisons)
    try write(report, to: options.out)
}

func write(_ value: some Encodable, to url: URL?) throws {
    guard let url else { return }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(value).write(to: url)
    print("wrote \(url.path)")
}

/// One arm from a copy of the checkpoint. The hold, when given, runs from the
/// first decoded token to the last.
func runArm(
    _ spec: ArmSpec, prompt: MLXArray, checkpoint: Checkpoint, parameters: GenerateParameters,
    model: any LanguageModel, drafter: any DFlash2DrafterModel, eos: Set<Int>,
    fallback: DFlash2Fallback, hold: GPUClockHold?
) throws -> Arm {
    let prefillTokens = prompt.dim(0) - checkpoint.prefix
    resetMemory()
    let start = now()
    defer { hold?.stop() }
    guard let block = spec.block else {
        var iterator = try TokenIterator(
            input: LMInput(tokens: prompt[checkpoint.prefix...]), model: model,
            cache: checkpoint.copied(), state: checkpoint.state, parameters: parameters)
        let result = measure(
            start: start, eos: eos, decoding: { hold?.start() }, next: { iterator.next() })
        hold?.stop()
        Stream.gpu.synchronize()
        return Arm(
            kind: spec.kind, block: nil, clockHold: spec.clockHold, tokens: result.tokens,
            prefillTokens: prefillTokens, prefillSeconds: result.prefill,
            decodeSeconds: result.decode)
    }
    var iterator = try DFlash2TokenIterator(
        input: LMInput(tokens: prompt), mainModel: model, drafter: drafter,
        mainCache: checkpoint.copied(), prefilledPrefixTokens: checkpoint.prefix,
        state: checkpoint.state, parameters: parameters, blockSize: block, fallback: fallback)
    var roundAccepts: [Int] = []
    var accepted = 0
    let result = measure(
        start: start, eos: eos, decoding: { hold?.start() },
        observe: {
            let rounds = iterator.speculativeDecodingTelemetry?.roundCount ?? 0
            if rounds > roundAccepts.count {
                roundAccepts.append(iterator.acceptedCount - accepted)
                accepted = iterator.acceptedCount
            }
        },
        next: { iterator.next() })
    hold?.stop()
    Stream.gpu.synchronize()
    return Arm(
        kind: spec.kind, block: block, clockHold: spec.clockHold, tokens: result.tokens,
        prefillTokens: prefillTokens, prefillSeconds: result.prefill,
        decodeSeconds: result.decode,
        rounds: iterator.speculativeDecodingTelemetry?.roundCount ?? 0,
        proposed: iterator.proposedCount, accepted: iterator.acceptedCount,
        roundAccepts: roundAccepts, passthroughReason: iterator.passthroughReason)
}

func summarize(_ turns: [Turn], options: Options) -> ([Summary], [HoldComparison]) {
    // One subset per `phase` the token files name, after the three every run has.
    let phases: [(String, (Turn) -> Bool)] = Set(turns.map(\.phase)).sorted().map { phase in
        (phase, { (turn: Turn) in turn.phase == phase })
    }
    let subsets: [(String, (Turn) -> Bool)] =
        [
            ("all", { _ in true }),
            ("long", { $0.arms.first { $0.kind == "plain" }!.outputTokens > 150 }),
            ("short", { $0.arms.first { $0.kind == "plain" }!.outputTokens <= 150 }),
        ] + phases
    func pooled(_ arms: [Arm]) -> (tokens: Int, seconds: Double, rate: Double, prefill: Double) {
        let tokens = arms.reduce(0) { $0 + $1.decodeTokens }
        let seconds = arms.reduce(0) { $0 + $1.decodeSeconds }
        let prefillTokens = arms.reduce(0) { $0 + $1.prefillTokens }
        let prefillSeconds = arms.reduce(0) { $0 + $1.prefillSeconds }
        return (tokens, seconds, Double(tokens) / seconds, Double(prefillTokens) / prefillSeconds)
    }
    let holds = options.clockHoldAB ? [false, true] : [options.clockHold]
    var summaries: [Summary] = []
    var comparisons: [HoldComparison] = []
    for temperature in options.temperatures {
        for (subset, filter) in subsets {
            let selected = turns.filter { $0.temperature == temperature && filter($0) }
            guard !selected.isEmpty else { continue }
            let arms = selected.flatMap(\.arms)
            for hold in holds {
                let plain = pooled(arms.filter { $0.kind == "plain" && $0.clockHold == hold })
                for block in options.blocks {
                    let spec = arms.filter { $0.block == block && $0.clockHold == hold }
                    let decode = pooled(spec)
                    let rounds = spec.reduce(0) { $0 + ($1.rounds ?? 0) }
                    let accepted = spec.reduce(0) { $0 + ($1.accepted ?? 0) }
                    let tau = Double(accepted + rounds) / Double(max(rounds, 1))
                    let roundCost =
                        (decode.seconds / Double(max(rounds, 1)))
                        / (plain.seconds / Double(max(plain.tokens, 1)))
                    summaries.append(
                        Summary(
                            temperature: temperature, subset: subset, clockHold: hold,
                            block: block, turns: selected.count, plainDecodeRate: plain.rate,
                            speculativeDecodeRate: decode.rate, speedup: decode.rate / plain.rate,
                            tau: tau, roundCost: roundCost, gain: tau / roundCost,
                            fellBack: spec.filter { $0.passthroughReason != nil }.count))
                }
            }
            guard options.clockHoldAB else { continue }
            for block in [nil] + options.blocks.map(Optional.init) {
                let kind = block == nil ? "plain" : "speculative"
                let on = pooled(
                    arms.filter { $0.kind == kind && $0.block == block && $0.clockHold })
                let off = pooled(
                    arms.filter { $0.kind == kind && $0.block == block && !$0.clockHold })
                comparisons.append(
                    HoldComparison(
                        temperature: temperature, subset: subset, kind: kind, block: block,
                        decodeRatio: on.rate / off.rate, prefillRatio: on.prefill / off.prefill))
            }
        }
    }
    return (summaries, comparisons)
}

func calibrationRow(_ calibration: DFlash2Calibration, seconds: Double?) -> String {
    String(
        format:
            "calibration block %d: R %.3f, plain %.2f ms, round %.2f ms, plain spread %.3f, plain drift %.3f, round spread %.3f, %d measurement(s)%@, %@",
        calibration.blockSize, calibration.roundCost, 1000 * calibration.plainStepSeconds,
        1000 * calibration.roundSeconds, calibration.plainSpread, calibration.plainDrift,
        calibration.roundSpread, calibration.measurements,
        (seconds.map { String(format: ", %.2f s", $0) } ?? "") as NSString,
        "\(calibration.outcome)" as NSString)
}

/// Every interval of a cold process's first decode: plain steps, rounds at the
/// first width, plain steps again, each from a fresh 32-token prompt.
func coldTrace(
    options: Options, model: any LanguageModel, drafter: any DFlash2DrafterModel
) throws {
    let parameters = GenerateParameters(temperature: options.temperatures[0], topP: options.topP)
    let span = 20_000
    let prompt = MLXArray((0 ..< 32).map { Int32((span / 4 + $0 * 7919) % span) })
    func plain(_ count: Int) throws -> [Double] {
        var iterator = try TokenIterator(
            input: LMInput(tokens: prompt), model: model, parameters: parameters)
        var stamps = [now()]
        for _ in 0 ... count {
            _ = iterator.next()
            stamps.append(now())
        }
        return zip(stamps.dropFirst(), stamps).map { 1000 * ($0 - $1) }
    }
    func rounds(_ count: Int) throws -> [Double] {
        var iterator = try DFlash2TokenIterator(
            input: LMInput(tokens: prompt), mainModel: model, drafter: drafter,
            parameters: parameters, blockSize: options.blocks[0], fallback: .disabled)
        var stamps = [now()]
        while (iterator.speculativeDecodingTelemetry?.roundCount ?? 0) < count {
            let before = iterator.speculativeDecodingTelemetry?.roundCount ?? 0
            _ = iterator.next()
            if (iterator.speculativeDecodingTelemetry?.roundCount ?? 0) > before {
                stamps.append(now())
            }
        }
        return zip(stamps.dropFirst(), stamps).map { 1000 * ($0 - $1) }
    }
    for (name, intervals) in [
        ("plain", try plain(60)), ("rounds \(options.blocks[0])", try rounds(40)),
        ("plain", try plain(40)),
    ] {
        print("\(name) ms: " + intervals.map { String(format: "%.1f", $0) }.joined(separator: " "))
    }
}

/// One cold prefill of a whole prompt.
struct PrefillArm: Codable {
    var stepSize: Int
    /// Whether forwards of `--prefill-dense-rows` rows or more took the dense route.
    var dense: Bool
    /// Whether the model's own linears were in place, not the swapped ones.
    var stock: Bool
    var seconds: Double
    var peakMemory: Int
    /// The last position's most likely token.
    var argmax: Int
    /// Largest absolute difference of the last position's logits within a pass: a dense
    /// arm's from the quantized arm's of the same ceiling, a quantized arm's from the
    /// quantized 512 ceiling's, which is what moving chunk boundaries alone costs.
    var logitDelta: Float
}

/// A decoder-layer quantized linear that multiplies by its dequantized weight when a
/// forward carries `minRows` rows or more: a prefill chunk, never a decode step or a
/// verify block.
final class DensePrefillLinear: QuantizedLinear {
    nonisolated(unsafe) static var minRows = Int.max

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard x.ndim >= 2, x.dim(-2) >= Self.minRows else { return super.callAsFunction(x) }
        let weight = dequantized(
            self.weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits,
            mode: mode, globalScale: globalScale, dtype: x.dtype)
        let result = matmul(x, weight.T)
        return bias.map { result + $0 } ?? result
    }
}

struct PrefillTurn: Codable {
    var label: String
    var phase: String
    var promptTokens: Int
    /// In run order.
    var arms: [PrefillArm]
}

/// One chunk ceiling pooled over every prompt: ratios of sums.
struct PrefillSummary: Codable {
    var stepSize: Int
    var dense: Bool
    var stock: Bool
    var tokens: Int
    var seconds: Double
    var rate: Double
    /// `rate` over the quantized 512 ceiling's, or over the first arm's when 512 is not swept.
    var ratio: Double
    var peakMemory: Int
    /// Largest `logitDelta` of any arm.
    var logitDelta: Float
    /// Arms whose most likely last token differs from the quantized arm's of the same ceiling.
    var argmaxMismatches: Int
}

/// Every decoder-layer quantized linear run alone on `rows` random rows.
struct KernelRows: Codable {
    var rows: Int
    /// Summed over the linears, through the quantized kernel.
    var quantizedSeconds: Double
    /// The same products as dense matmuls over weights dequantized beforehand.
    var denseSeconds: Double
    /// `quantizedSeconds` by the linear's name in its layer.
    var quantizedSecondsByKind: [String: Double]
    /// Every gated-delta mixer whole, from zero state: its projections, which
    /// `quantizedSeconds` also counts, plus the convolution, the scan and the norm.
    var gatedDeltaSeconds: Double
}

struct KernelProfile: Codable {
    var linears: Int
    var parameters: Int
    var gatedDeltaLayers: Int
    var attentionLayers: Int
    /// Dequantizing every weight once, which a dense route would pay per chunk.
    var dequantizeSeconds: Double
    var rows: [KernelRows]
}

struct PrefillReport: Codable {
    var target: String
    var turns: [PrefillTurn]
    var summaries: [PrefillSummary]
    var profile: KernelProfile?
}

func prefillStudy(options: Options, model: any LanguageModel) throws {
    var turns: [PrefillTurn] = []
    var summaries: [PrefillSummary] = []
    if let sizes = options.prefillSweep {
        /// (dense, stock) per arm of one ceiling.
        var routes = [(false, false)]
        var install: (Bool) -> Void = { _ in }
        if let rows = options.prefillDenseRows {
            guard let module = model as? Module else {
                throw BenchError.usage("the target is no Module")
            }
            var originals: [(String, Module)] = []
            let swaps = module.namedModules().compactMap { name, child -> (String, Module)? in
                // The gated-delta input projections stay fused, as the load left them:
                // replacing one drops the fusion for every forward.
                guard name.contains(".layers."), !name.contains("vision"),
                    !name.contains(".linear_attn.in_proj_"),
                    let linear = child as? QuantizedLinear
                else { return nil }
                originals.append((name, linear))
                return (
                    name,
                    DensePrefillLinear(
                        weight: linear.weight, bias: linear.bias, scales: linear.scales,
                        biases: linear.biases, groupSize: linear.groupSize, bits: linear.bits,
                        mode: linear.mode, globalScale: linear.globalScale)
                )
            }
            let stock = originals
            install = { useStock in
                module.update(modules: ModuleChildren.unflattened(useStock ? stock : swaps))
            }
            routes =
                (options.prefillStockArm ? [(false, true)] : []) + [(false, false), (true, false)]
            print("dense route: \(swaps.count) linears, forwards of \(rows) rows or more")
        }
        func coldPrefill(_ prompt: MLXArray, stepSize: Int, dense: Bool, stock: Bool) throws
            -> (PrefillArm, MLXArray)
        {
            install(stock)
            DensePrefillLinear.minRows = dense ? options.prefillDenseRows ?? .max : .max
            defer { DensePrefillLinear.minRows = .max }
            resetMemory()
            let cache = try model.newCache(parameters: nil)
            let start = now()
            guard
                case .logits(let output) = try model.prepare(
                    LMInput(tokens: prompt), cache: cache, state: nil,
                    prefill: .init(stepSize: stepSize))
            else { throw BenchError.usage("prepare left tokens") }
            eval(cache.flatMap { $0.innerState() } + [output.logits])
            let seconds = now() - start
            let last = output.logits[0, -1].asType(.float32)
            eval(last)
            return (
                PrefillArm(
                    stepSize: stepSize, dense: dense, stock: stock, seconds: seconds,
                    peakMemory: Memory.peakMemory, argmax: argMax(last).item(Int.self),
                    logitDelta: 0),
                last
            )
        }
        let names =
            try options.files.map { $0.map { "\($0).json" } }
            ?? FileManager.default.contentsOfDirectory(atPath: options.tokens.path)
            .filter { $0.hasSuffix(".json") }.sorted()
        let order = sizes.flatMap { size in routes.map { (size, $0.0, $0.1) } }
        var warm = false
        for name in names {
            let file = try JSONDecoder().decode(
                TokenFile.self, from: Data(contentsOf: options.tokens.appending(component: name)))
            let prompt = MLXArray(file.tokenIds.map(Int32.init))
            if !warm {
                // Compiles each arm's kernels before anything is timed.
                for (size, dense, stock) in order {
                    _ = try coldPrefill(
                        prompt[..<min(prompt.dim(0), 2 * size + 1)], stepSize: size, dense: dense,
                        stock: stock)
                }
                warm = true
            }
            var arms: [PrefillArm] = []
            for pass in [order, order.reversed()] {
                var quantized: [Int: MLXArray] = [:]
                var pending: [(Int, MLXArray)] = []
                for (size, dense, stock) in pass {
                    let (arm, last) = try coldPrefill(
                        prompt, stepSize: size, dense: dense, stock: stock)
                    pending.append((arms.count, last))
                    if !dense, !stock { quantized[size] = last }
                    arms.append(arm)
                    print(
                        String(
                            format: "%-22@ chunk %5d %@  %7.2f s  %7.1f tok/s  peak %.0f MiB",
                            file.label as NSString, size,
                            (stock ? "stock" : dense ? "dense" : "quant") as NSString,
                            arm.seconds, Double(file.tokenIds.count) / arm.seconds,
                            Double(arm.peakMemory) / 1_048_576))
                    fflush(stdout)
                }
                for (index, last) in pending {
                    let arm = arms[index]
                    guard
                        let reference = quantized[
                            arm.dense || arm.stock
                                ? arm.stepSize : PrefillParameters.defaultStepSize]
                    else { continue }
                    arms[index].logitDelta = abs(last - reference).max().item(Float.self)
                }
            }
            turns.append(
                PrefillTurn(
                    label: file.label, phase: file.phase, promptTokens: file.tokenIds.count,
                    arms: arms))
        }
        let pooled = order.map { size, dense, stock -> PrefillSummary in
            var tokens = 0
            var seconds = 0.0
            var peak = 0
            var delta: Float = 0
            var mismatches = 0
            for turn in turns {
                for arm in turn.arms
                where arm.stepSize == size && arm.dense == dense && arm.stock == stock {
                    tokens += turn.promptTokens
                    seconds += arm.seconds
                    peak = max(peak, arm.peakMemory)
                    delta = max(delta, arm.logitDelta)
                    let reference = turn.arms.first {
                        $0.stepSize == size && !$0.dense && !$0.stock
                    }
                    mismatches += reference?.argmax == arm.argmax ? 0 : 1
                }
            }
            return PrefillSummary(
                stepSize: size, dense: dense, stock: stock, tokens: tokens, seconds: seconds,
                rate: Double(tokens) / seconds, ratio: 1, peakMemory: peak, logitDelta: delta,
                argmaxMismatches: mismatches)
        }
        let base =
            (pooled.first {
                $0.stepSize == PrefillParameters.defaultStepSize && !$0.dense && !$0.stock
            } ?? pooled[0]).rate
        summaries = pooled.map {
            var summary = $0
            summary.ratio = $0.rate / base
            return summary
        }
        print("")
        print("chunk route   tok/s   ratio  peak MiB  logit delta  argmax differs")
        for summary in summaries {
            print(
                String(
                    format: "%5d %@  %6.1f  %5.3f  %8.0f  %11.4f  %d", summary.stepSize,
                    (summary.stock ? "stock" : summary.dense ? "dense" : "quant") as NSString,
                    summary.rate, summary.ratio,
                    Double(summary.peakMemory) / 1_048_576, summary.logitDelta,
                    summary.argmaxMismatches))
        }
    }
    let profile = options.prefillProfile ? try kernelProfile(options: options, model: model) : nil
    try write(
        PrefillReport(
            target: options.target.path, turns: turns, summaries: summaries, profile: profile),
        to: options.out)
}

func kernelProfile(options: Options, model: any LanguageModel) throws -> KernelProfile {
    guard let module = model as? Module else { throw BenchError.usage("the target is no Module") }
    let linears = module.namedModules().compactMap { name, child in
        name.contains(".layers.") && !name.contains("vision")
            ? (child as? QuantizedLinear).map { (name, $0) } : nil
    }
    let parameters = linears.reduce(0) { $0 + $1.1.shape.0 * $1.1.shape.1 }
    let mixers = module.namedModules().compactMap { name, child in
        (child as? Qwen35Language.GatedDeltaNet).map { (name, $0) }
    }
    let attentionLayers = module.namedModules().filter {
        $0.0.contains(".layers.") && $0.0.hasSuffix(".self_attn")
    }.count
    print("")
    print(
        "kernel profile: \(linears.count) quantized linears, \(parameters) parameters, "
            + "\(mixers.count) gated-delta layers, \(attentionLayers) attention layers")

    func timed(_ array: MLXArray) -> Double {
        let start = now()
        eval(array)
        return now() - start
    }
    func dense(_ linear: QuantizedLinear) -> MLXArray {
        dequantized(
            linear.weight, scales: linear.scales, biases: linear.biases,
            groupSize: linear.groupSize, bits: linear.bits, mode: linear.mode,
            globalScale: linear.globalScale)
    }

    var dequantizeSeconds = 0.0
    var rows: [KernelRows] = []
    for (index, count) in ([options.profileRows[0]] + options.profileRows).enumerated() {
        var entry = KernelRows(
            rows: count, quantizedSeconds: 0, denseSeconds: 0, quantizedSecondsByKind: [:],
            gatedDeltaSeconds: 0)
        var dequantize = 0.0
        for (name, linear) in linears {
            autoreleasepool {
                let x = MLXRandom.normal([1, count, linear.shape.1]).asType(linear.scales.dtype)
                eval(x)
                let quantized = timed(linear(x))
                let weight = dense(linear)
                dequantize += timed(weight)
                let product = timed(matmul(x, weight.T))
                entry.quantizedSeconds += quantized
                entry.denseSeconds += product
                let kind = String(name.split(separator: ".").last ?? "")
                entry.quantizedSecondsByKind[kind, default: 0] += quantized
            }
            Memory.clearCache()
        }
        for (name, mixer) in mixers {
            guard let projection = linears.first(where: { $0.0 == name + ".out_proj" })?.1
            else { throw BenchError.usage("\(name) has no quantized out_proj") }
            autoreleasepool {
                let x = MLXRandom.normal([1, count, projection.shape.0]).asType(
                    projection.scales.dtype)
                eval(x)
                entry.gatedDeltaSeconds += timed(mixer(x))
            }
            Memory.clearCache()
        }
        // The first pass over the first row count compiles kernels and is dropped.
        guard index > 0 else { continue }
        dequantizeSeconds = index == 1 ? dequantize : min(dequantizeSeconds, dequantize)
        rows.append(entry)
        print(
            String(
                format:
                    "rows %5d  quantized %8.3f s (%7.1f rows/s)  dense %8.3f s (%7.1f rows/s)  dequantize %.3f s  gated-delta %.3f s",
                count, entry.quantizedSeconds, Double(count) / entry.quantizedSeconds,
                entry.denseSeconds, Double(count) / entry.denseSeconds, dequantize,
                entry.gatedDeltaSeconds))
        print(
            "  by kind: "
                + entry.quantizedSecondsByKind.sorted { $0.key < $1.key }.map {
                    String(format: "%@ %.3f", $0.key as NSString, $0.value)
                }.joined(separator: "  "))
        fflush(stdout)
    }
    return KernelProfile(
        linears: linears.count, parameters: parameters, gatedDeltaLayers: mixers.count,
        attentionLayers: attentionLayers, dequantizeSeconds: dequantizeSeconds, rows: rows)
}

/// One width of the cost sweep.
struct SweepWidth: Codable {
    var calibration: Calibration
    /// Median milliseconds of each phase behind barriers.
    var phaseMilliseconds: [String: Double]
    var phaseSumMilliseconds: Double
    var barrieredRounds: Int
}

struct SweepReport: Codable {
    var target: String
    var drafter: String
    var clockHold: Bool
    var widths: [SweepWidth]
}

func costSweep(
    options: Options, model: any LanguageModel, drafter: any DFlash2DrafterModel,
    hold: GPUClockHold?
) throws {
    let parameters = GenerateParameters(temperature: options.temperatures[0], topP: options.topP)
    let span = 20_000
    let prompt = MLXArray((0 ..< 32).map { Int32((span / 4 + $0 * 7919) % span) })
    let warmup = 2
    let timed = 16
    var widths: [SweepWidth] = []
    for width in options.sweepWidths {
        if options.clockHold { hold?.start() }
        let calibration = try dflash2Calibrate(
            target: model, drafter: drafter, blockSize: width, parameters: parameters)

        let timer = DFlash2PhaseTimer()
        var iterator = try DFlash2TokenIterator(
            input: LMInput(tokens: prompt), mainModel: model, drafter: drafter,
            parameters: parameters, blockSize: width, fallback: .disabled)
        iterator.phaseTimer = timer
        while (iterator.speculativeDecodingTelemetry?.roundCount ?? 0) < warmup {
            _ = iterator.next()
        }
        timer.reset()
        while (iterator.speculativeDecodingTelemetry?.roundCount ?? 0) < warmup + timed {
            _ = iterator.next()
        }
        hold?.stop()
        var phases: [String: Double] = [:]
        for phase in DFlash2PhaseTimer.Phase.allCases {
            phases[phase.rawValue] = 1000 * medianOf(timer.seconds[phase] ?? [0])
        }
        let sweep = SweepWidth(
            calibration: Calibration(calibration), phaseMilliseconds: phases,
            phaseSumMilliseconds: phases.values.reduce(0, +), barrieredRounds: timed)
        widths.append(sweep)
        print(sweepRow(sweep))
        fflush(stdout)
    }
    print("")
    print(sweepHeader)
    for width in widths { print(sweepRow(width)) }
    try write(
        SweepReport(
            target: options.target.path, drafter: options.drafter.path,
            clockHold: options.clockHold, widths: widths),
        to: options.out)
}

let sweepHeader =
    "width  R      plain ms  round ms  spread  | barriered ms: draft  verify  replay  sampling  commit  sum    | replay+sampling  self-check"

func sweepRow(_ sweep: SweepWidth) -> String {
    let p = sweep.phaseMilliseconds
    let share = (p["replay"]! + p["sampling"]!) / sweep.phaseSumMilliseconds
    return String(
        format:
            "%-6d %-6.3f %-9.2f %-9.2f %-7.3f |               %-6.2f %-7.2f %-7.2f %-9.2f %-7.2f %-6.2f | %-16.3f %@",
        sweep.calibration.block, sweep.calibration.roundCost,
        1000 * sweep.calibration.plainStepSeconds, 1000 * sweep.calibration.roundSeconds,
        sweep.calibration.plainSpread, p["draft"]!, p["verify"]!, p["replay"]!, p["sampling"]!,
        p["commit"]!, sweep.phaseSumMilliseconds, share, sweep.calibration.outcome as NSString)
}

/// Two-sided Mann-Whitney U p-value, normal approximation with tie correction.
func mannWhitney(_ a: [Int], _ b: [Int]) -> Double {
    let pooled = (a.map { (Double($0), 0) } + b.map { (Double($0), 1) }).sorted { $0.0 < $1.0 }
    var ranks = [Double](repeating: 0, count: pooled.count)
    var tieTerm = 0.0
    var i = 0
    while i < pooled.count {
        var j = i
        while j + 1 < pooled.count, pooled[j + 1].0 == pooled[i].0 { j += 1 }
        let rank = Double(i + j) / 2 + 1
        for k in i ... j { ranks[k] = rank }
        let t = Double(j - i + 1)
        tieTerm += t * t * t - t
        i = j + 1
    }
    let n1 = Double(a.count)
    let n2 = Double(b.count)
    let r1 = zip(pooled, ranks).filter { $0.0.1 == 0 }.reduce(0) { $0 + $1.1 }
    let u = r1 - n1 * (n1 + 1) / 2
    let mean = n1 * n2 / 2
    let n = n1 + n2
    let variance = n1 * n2 / 12 * ((n + 1) - tieTerm / (n * (n - 1)))
    guard variance > 0 else { return 1 }
    let z = abs(u - mean) / variance.squareRoot()
    return erfc(z / 2.0.squareRoot())
}

func median(_ values: [Int]) -> Double {
    let sorted = values.sorted()
    let middle = sorted.count / 2
    return sorted.count % 2 == 0
        ? Double(sorted[middle - 1] + sorted[middle]) / 2 : Double(sorted[middle])
}

func lengthStudy(
    options: Options, seeds: Int, model: any LanguageModel, drafter: any DFlash2DrafterModel,
    eos: Set<Int>
) throws {
    let temperature = options.temperatures[0]
    for name in options.files ?? [] {
        let file = try JSONDecoder().decode(
            TokenFile.self,
            from: Data(contentsOf: options.tokens.appending(component: "\(name).json")))
        let prompt = MLXArray(file.tokenIds.map(Int32.init))
        var outputs: [String: [[Int]]] = [:]
        var taus: [Double] = []
        var arms = ["tokenIterator", "speculative", "seededStep", "seededVerifyRoute"]
        var checkpoint: (cache: [KVCache], state: LMOutput.State?)? = nil
        if let prefix = options.warmPrefix {
            arms = ["tokenIterator", "speculative"]
            let cache = try model.newCache(parameters: nil)
            guard
                case .logits(let output) = try model.prepare(
                    LMInput(tokens: prompt[..<prefix]), cache: cache, state: nil,
                    prefill: .init())
            else { throw BenchError.usage("prepare left tokens") }
            eval(cache)
            checkpoint = (cache, output.state)
            print("warm prefix \(prefix) of \(file.tokenIds.count)")
        }
        func copied() -> [KVCache]? { checkpoint.map { $0.cache.map { $0.copy() } } }
        let prefix = options.warmPrefix ?? 0
        for seed in 0 ..< seeds {
            var parameters = GenerateParameters(
                maxTokens: options.maxTokens, temperature: temperature, topP: options.topP)
            parameters.seed = UInt64(seed)
            for arm in arms {
                resetMemory()
                let tokens: [Int]
                switch arm {
                case "tokenIterator":
                    var iterator = try TokenIterator(
                        input: LMInput(tokens: prompt[prefix...]), model: model,
                        cache: copied(), state: checkpoint?.state, parameters: parameters)
                    tokens = measure(start: now(), eos: eos) { iterator.next() }.tokens
                case "speculative":
                    var iterator = try DFlash2TokenIterator(
                        input: LMInput(tokens: prompt), mainModel: model, drafter: drafter,
                        mainCache: copied(), prefilledPrefixTokens: prefix,
                        state: checkpoint?.state, parameters: parameters,
                        blockSize: options.blocks[0], fallback: .disabled)
                    tokens = measure(start: now(), eos: eos) { iterator.next() }.tokens
                    if let telemetry = iterator.speculativeDecodingTelemetry {
                        taus.append(
                            Double(telemetry.acceptedDraftTokenCount + telemetry.roundCount)
                                / Double(telemetry.roundCount))
                    }
                default:
                    var iterator = try DFlash2TokenIterator(
                        input: LMInput(tokens: prompt), mainModel: model, drafter: drafter,
                        parameters: parameters, blockSize: options.blocks[0], speculates: false,
                        plainStepsVerify: arm == "seededVerifyRoute")
                    tokens = measure(start: now(), eos: eos) { iterator.next() }.tokens
                }
                Stream.gpu.synchronize()
                outputs[arm, default: []].append(tokens)
            }
            print(
                "\(file.label) seed \(seed): "
                    + arms.map { "\($0)=\(outputs[$0]!.last!.count)" }.joined(separator: " "))
            fflush(stdout)
        }
        let lengths = outputs.mapValues { $0.map(\.count) }
        print("\n\(file.label) T=\(temperature): output tokens over \(seeds) seeds")
        for arm in arms {
            let values = lengths[arm]!
            print(
                String(
                    format: "  %-18@ median %6.1f  mean %7.1f  lengths %@", arm as NSString,
                    median(values), Double(values.reduce(0, +)) / Double(values.count),
                    values.description as NSString))
        }
        for (a, b) in [
            ("tokenIterator", "speculative"), ("tokenIterator", "seededStep"),
            ("speculative", "seededStep"), ("speculative", "seededVerifyRoute"),
            ("seededStep", "seededVerifyRoute"), ("tokenIterator", "seededVerifyRoute"),
        ] where lengths[a] != nil && lengths[b] != nil {
            print(
                String(
                    format: "  Mann-Whitney %@ vs %@: p=%.4f", a as NSString, b as NSString,
                    mannWhitney(lengths[a]!, lengths[b]!)))
        }
        // The three seeded arms share per-position keys, so per seed they differ
        // only in numerics.
        for (a, b) in [
            ("speculative", "seededStep"), ("speculative", "seededVerifyRoute"),
            ("seededStep", "seededVerifyRoute"),
        ] where outputs[a] != nil && outputs[b] != nil {
            let pairs = zip(outputs[a]!, outputs[b]!)
            let identical = pairs.filter { $0.0 == $0.1 }.count
            let divergences = pairs.compactMap { x, y in
                zip(x, y).enumerated().first { $0.element.0 != $0.element.1 }?.offset
            }
            print(
                "  paired \(a) vs \(b): identical \(identical)/\(seeds), "
                    + "median first divergence \(divergences.isEmpty ? 0 : median(divergences))")
        }
        print(
            String(
                format: "  speculative mean tau %.2f",
                taus.reduce(0, +) / Double(max(taus.count, 1))))
    }
}

func row(_ label: String, _ temperature: Float, _ arm: Arm) -> String {
    let name = arm.block.map { "spec \($0)" } ?? "plain"
    return String(
        format:
            "%-22@ T=%.2f %-8@ hold %-3@ prefill %6.0f tok/s  decode %5.1f tok/s  out %4d  tau %.2f  rounds %4d  peak %.0f MiB%@",
        label as NSString, temperature, name as NSString,
        (arm.clockHold ? "on" : "off") as NSString, arm.prefillRate, arm.decodeRate,
        arm.outputTokens, arm.tau ?? 0, arm.rounds ?? 0, Double(arm.peakMemory) / 1_048_576,
        (arm.passthroughReason.map { "  fell back: \($0)" } ?? "") as NSString)
}

@main
struct DFlash2Bench {
    static func main() async throws {
        try await run(Options(CommandLine.arguments))
    }
}
