import {
    contextWindowDetectionSchema,
    parseResponse,
    type ContextWindowDetection
} from '$lib/schemas';

/** The model identifier sent to the context-window detection endpoint. */
export interface ContextWindowDetectionRequest {
    model: string;
}

/**
 * Fetch a model's context-window detection result.
 *
 * @param apiBase The configured backend origin.
 * @param request The model identifier to inspect.
 * @returns The validated detection result.
 * @throws {Error} If the backend cannot serve the detection request.
 */
export async function fetchContextWindowDetection(
    apiBase: string,
    request: ContextWindowDetectionRequest
): Promise<ContextWindowDetection> {
    const response: Response = await fetch(
        `${apiBase}/api/v1/ollama/context-window?model=${encodeURIComponent(request.model)}`
    );
    if (!response.ok) throw new Error(`The server returned ${response.status}`);

    const body: unknown = await response.json();
    return parseResponse(contextWindowDetectionSchema, body, 'context window detection');
}
