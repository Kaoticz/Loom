import { z } from 'zod';

/**
 * Runtime schemas for every piece of data the frontend receives from outside itself.
 *
 * These schemas are the single source of truth for these shapes: the TypeScript
 * types the rest of the application uses are derived from them with `z.infer`
 * rather than declared separately, so a type and its validation cannot drift
 * apart.
 *
 * Objects strip properties they do not describe rather than rejecting them, so a
 * newer backend that adds a field does not break an older frontend.
 */

/** Where a tool call currently stands. */
export const toolStatusSchema = z.enum(['running', 'success', 'failure']);

const thoughtMessageSchema = z.object({
    type: z.literal('thought'),
    content: z.string(),
    stream_id: z.string().nullish()
});

const planMessageSchema = z.object({
    type: z.literal('plan'),
    steps: z.array(z.string()).default([]),
    requires_approval: z.boolean().default(true)
});

const actionMessageSchema = z.object({
    type: z.literal('action'),
    tool_name: z.string(),
    arguments: z.record(z.unknown()).default({}),
    status: toolStatusSchema.default('running')
});

const observationMessageSchema = z.object({
    type: z.literal('observation'),
    tool_name: z.string(),
    result: z.string(),
    status: toolStatusSchema.default('success')
});

const commitMessageSchema = z.object({
    type: z.literal('commit'),
    commit_hash: z.string(),
    message: z.string(),
    files_changed: z.array(z.string()).default([])
});

const diffMessageSchema = z.object({
    type: z.literal('diff'),
    file_path: z.string(),
    original: z.string(),
    modified: z.string(),
    truncated: z.boolean().default(false)
});

const reviewMessageSchema = z.object({
    type: z.literal('review'),
    approved: z.boolean(),
    comments: z.string()
});

const doneMessageSchema = z.object({
    type: z.literal('done'),
    summary: z.string(),
    branch_name: z.string().default(''),
    total_commits: z.number().int().nonnegative().default(0)
});

const errorMessageSchema = z.object({
    type: z.literal('error'),
    content: z.string()
});

const sessionBusyMessageSchema = z.object({
    type: z.literal('session_busy'),
    content: z.string(),
    active_session_id: z.string().default('')
});

/**
 * The phase of work a session announces it has entered.
 *
 * Stated by the backend rather than inferred by the client from the wording of
 * announcements, which breaks silently whenever the wording changes.
 */
export const sessionStatusSchema = z.enum([
    'indexing',
    'planning',
    'executing',
    'reviewing',
    're_working',
    'merging',
    'cancelled'
]);

const systemMessageSchema = z.object({
    type: z.literal('system'),
    content: z.string(),
    status: sessionStatusSchema.nullish(),
    key: z.string().nullish(),
    processed: z.number().int().nonnegative().nullish(),
    total: z.number().int().nonnegative().nullish()
});

/**
 * Which step of preparing the sandbox failed.
 *
 * Every cause used to arrive as one indistinguishable sentence, which is what
 * made an unusable sandbox hard to diagnose.
 */
export const sandboxStageSchema = z.enum([
    'runtime_connectivity',
    'path_discovery',
    'image_build',
    'image_pull',
    'container_start',
    'readiness_probe'
]);

const sandboxUnavailableMessageSchema = z.object({
    type: z.literal('sandbox_unavailable'),
    stage: sandboxStageSchema,
    reason: z.string()
});

/** Every message the agent WebSocket can deliver. */
export const agentMessageSchema = z.discriminatedUnion('type', [
    thoughtMessageSchema,
    planMessageSchema,
    actionMessageSchema,
    observationMessageSchema,
    commitMessageSchema,
    diffMessageSchema,
    reviewMessageSchema,
    doneMessageSchema,
    errorMessageSchema,
    sessionBusyMessageSchema,
    systemMessageSchema,
    sandboxUnavailableMessageSchema
]);

export type AgentMessagePayload = z.infer<typeof agentMessageSchema>;
export type ThoughtMessagePayload = z.infer<typeof thoughtMessageSchema>;
export type SystemMessagePayload = z.infer<typeof systemMessageSchema>;
export type SandboxUnavailablePayload = z.infer<typeof sandboxUnavailableMessageSchema>;
export type SandboxStage = z.infer<typeof sandboxStageSchema>;
export type ActionMessagePayload = z.infer<typeof actionMessageSchema>;

/** Lifecycle status of a task, as reported over the REST API. */
export const taskStatusSchema = z.enum([
    'initialized',
    'running',
    'completed',
    'merged',
    'failed',
    'cancelled'
]);

/** The kind of event a task log entry records. */
export const taskLogEntryTypeSchema = z.enum([
    'tool_call',
    'commit',
    'review',
    'plan',
    'rejection',
    'status_change',
    'summary'
]);

export const taskSchema = z.object({
    id: z.string(),
    status: taskStatusSchema,
    prompt: z.string(),
    repo_path: z.string(),
    branch_name: z.string(),
    summary: z.string(),
    created_at: z.string(),
    updated_at: z.string()
});

export const taskLogEntrySchema = z.object({
    entry_type: taskLogEntryTypeSchema,
    content: z.record(z.unknown()).default({}),
    created_at: z.string()
});

export const taskListSchema = z.object({
    items: z.array(taskSchema).default([]),
    total: z.number().int().nonnegative(),
    offset: z.number().int().nonnegative(),
    limit: z.number().int().positive()
});

export const taskDetailSchema = z.object({
    task: taskSchema,
    coder_model: z.string(),
    reviewer_model: z.string(),
    log: z.array(taskLogEntrySchema).default([])
});

/** The models Ollama reports as available. */
export const ollamaModelsSchema = z.array(z.string());

/**
 * The commands the client may send the session.
 *
 * Outbound data is not validated on arrival anywhere, so without this the set
 * would exist only in the backend and drift here unnoticed. Declaring it lets
 * the protocol guard compare the two.
 */
export const commandActionSchema = z.enum([
    'approve',
    'reject',
    'merge',
    'abort',
    'stop',
    'continue',
    'auto_approve'
]);

export type CommandAction = z.infer<typeof commandActionSchema>;

/** The verdict on whether a path is a usable git repository. */
export const repoValidationSchema = z.object({
    valid: z.boolean()
});

/** The subdirectory names of a path. */
export const directoryListingSchema = z.array(z.string());

export type Task = z.infer<typeof taskSchema>;
export type TaskStatus = z.infer<typeof taskStatusSchema>;
export type TaskLogEntry = z.infer<typeof taskLogEntrySchema>;
export type TaskList = z.infer<typeof taskListSchema>;
export type TaskDetail = z.infer<typeof taskDetailSchema>;

/**
 * Raised when data from outside the frontend does not match its schema.
 *
 * Carrying a distinct type lets a view report validation as the cause rather
 * than reporting the same generic failure it would show for a network error.
 */
export class SchemaValidationError extends Error {
    readonly issues: string;

    constructor(subject: string, error: z.ZodError) {
        const issues = error.issues
            .map((issue) => `${issue.path.join('.') || '(root)'}: ${issue.message}`)
            .join('; ');
        super(`The ${subject} from the server did not match the expected format (${issues}).`);
        this.name = 'SchemaValidationError';
        this.issues = issues;
    }
}

/**
 * Validate a decoded response body, naming the subject in any failure.
 *
 * @param schema The schema the body must satisfy.
 * @param body The decoded response body.
 * @param subject What was being fetched, used in the error message.
 * @returns The validated value, with any schema defaults applied.
 * @throws SchemaValidationError If the body does not match the schema.
 */
export function parseResponse<S extends z.ZodTypeAny>(
    schema: S,
    body: unknown,
    subject: string
): z.output<S> {
    const result = schema.safeParse(body);
    if (result.success) return result.data;
    throw new SchemaValidationError(subject, result.error);
}
