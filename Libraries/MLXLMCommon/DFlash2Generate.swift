import Foundation
import MLX

/// Generates text with DFlash2 block speculative decoding.
///
/// Parallel to the MTP ``generate(input:cache:parameters:context:mtpDrafter:blockSize:components:wiredMemoryTicket:tools:)``,
/// with two differences a cached prompt needs: `input` is the whole rendered
/// prompt, of which `cache` already holds the first `prefilledPrefixTokens`
/// positions, and `state` is the target state that cache continues from.
/// Only the rest is prefilled. The drafter shares the target's tokenizer.
///
/// - Parameters:
///   - input: the whole prompt, text only.
///   - cache: a target cache holding `prefilledPrefixTokens` positions; a new
///     one when nil.
///   - prefilledPrefixTokens: leading prompt positions `cache` already holds.
///   - state: the target's state for `cache`.
///   - parameters: sampling, prefill and generation limits.
///   - context: model context for the target.
///   - dflash2Drafter: the drafter. It holds no per-stream state, so one
///     instance can serve many streams.
///   - blockSize: tokens per verify pass; the width `fallback`'s round cost
///     was measured at.
///   - fallback: when a turn stops speculating and finishes with plain steps;
///     carries the round cost ``dflash2Calibrate(target:drafter:blockSize:parameters:)``
///     measured on this host at `blockSize`.
///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``.
///   - wiredMemoryTicket: optional wired memory ticket.
///   - tools: optional tool schemas used to parse arguments and authorize function names.
/// - Returns: the stream and the task that owns the iterator. A caller that stops
///   consuming early should cancel the task and await it before reusing `cache`:
///   the task ends once the in-flight round is finished and rewound.
/// - Throws: ``DFlash2SpeculationError`` when the target, cache, state, prompt
///   or round cost cannot be speculated over; nothing has been prefilled then.
public func generate(
    input: LMInput,
    cache: [KVCache]? = nil,
    prefilledPrefixTokens: Int = 0,
    state: LMOutput.State? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    dflash2Drafter: any DFlash2DrafterModel,
    blockSize: Int,
    fallback: DFlash2Fallback,
    components: GenerationComponents = .init(),
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    tools: [[String: any Sendable]]? = nil
) throws -> (AsyncStream<Generation>, Task<Void, Never>) {
    // Read the prompt before the iterator queues its prefill, which this would wait for.
    let reasoningPrimedInside = promptPrimesReasoning(
        input: input, modelConfiguration: context.configuration, tokenizer: context.tokenizer)
    let iterator = try DFlash2TokenIterator(
        input: input,
        mainModel: context.model,
        drafter: dflash2Drafter,
        mainCache: cache,
        prefilledPrefixTokens: prefilledPrefixTokens,
        state: state,
        parameters: parameters,
        blockSize: blockSize,
        fallback: fallback,
        components: components
    )
    var handler = TextToolTokenLoopHandler(
        tokenizer: context.tokenizer,
        stopStrings: context.configuration.effectiveStopStrings,
        format: context.configuration.toolCallFormat ?? .json,
        tools: tools,
        toolCallPolicy: parameters.toolCallPolicy,
        reasoning: context.configuration.reasoningConfig.map {
            (config: $0, primedInside: reasoningPrimedInside)
        }
    )
    handler.cachedPromptTokenCount = prefilledPrefixTokens
    return generateLoopTask(
        promptTokenCount: input.text.tokens.size - prefilledPrefixTokens,
        modelConfiguration: context.configuration,
        tokenizer: context.tokenizer,
        iterator: iterator,
        wiredMemoryTicket: wiredMemoryTicket,
        handler: handler
    )
}

/// ``generate(input:cache:prefilledPrefixTokens:state:parameters:context:dflash2Drafter:blockSize:fallback:components:wiredMemoryTicket:tools:)``
/// with the drafter read from its container, for a caller inside a
/// ``ModelContainer/perform(_:)`` closure.
public func generate(
    input: LMInput,
    cache: [KVCache]? = nil,
    prefilledPrefixTokens: Int = 0,
    state: LMOutput.State? = nil,
    parameters: GenerateParameters,
    context: ModelContext,
    dflash2Drafter: DFlash2DrafterContainer,
    blockSize: Int,
    fallback: DFlash2Fallback,
    components: GenerationComponents = .init(),
    wiredMemoryTicket: WiredMemoryTicket? = nil,
    tools: [[String: any Sendable]]? = nil
) async throws -> (AsyncStream<Generation>, Task<Void, Never>) {
    let drafter = await dflash2Drafter.perform { SendableBox($0.model) }.consume()
    return try generate(
        input: input, cache: cache, prefilledPrefixTokens: prefilledPrefixTokens, state: state,
        parameters: parameters, context: context, dflash2Drafter: drafter, blockSize: blockSize,
        fallback: fallback, components: components, wiredMemoryTicket: wiredMemoryTicket,
        tools: tools)
}
