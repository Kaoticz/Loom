import { describe, expect, it } from 'vitest';
import {
    agentMessageSchema,
    directoryListingSchema,
    ollamaModelsSchema,
    parseResponse,
    repoValidationSchema,
    SchemaValidationError,
    taskDetailSchema,
    taskListSchema
} from './schemas';

describe('parseResponse', () => {
    it('returns the validated value', () => {
        const models = parseResponse(ollamaModelsSchema, ['llama3', 'qwen'], 'model list');

        expect(models).toEqual(['llama3', 'qwen']);
    });

    it('throws a distinguishable error naming the subject', () => {
        expect(() => parseResponse(ollamaModelsSchema, { nope: true }, 'model list')).toThrow(
            SchemaValidationError
        );
        expect(() => parseResponse(ollamaModelsSchema, { nope: true }, 'model list')).toThrow(
            /model list/
        );
    });

    it('describes which field was wrong', () => {
        try {
            parseResponse(repoValidationSchema, { valid: 'yes' }, 'repository check');
            expect.unreachable('validation should have failed');
        } catch (error) {
            expect(error).toBeInstanceOf(SchemaValidationError);
            expect((error as SchemaValidationError).issues).toContain('valid');
        }
    });

    it('applies defaults for fields the backend omitted', () => {
        const list = parseResponse(taskListSchema, { total: 0, offset: 0, limit: 20 }, 'task list');

        expect(list.items).toEqual([]);
    });
});

describe('forward compatibility', () => {
    it('ignores unknown fields on a REST response rather than rejecting it', () => {
        const validation = parseResponse(
            repoValidationSchema,
            { valid: true, added_by_a_newer_backend: 'ignore me' },
            'repository check'
        );

        expect(validation).toEqual({ valid: true });
    });

    it('ignores unknown fields on an agent message', () => {
        const result = agentMessageSchema.safeParse({
            type: 'review',
            approved: true,
            comments: 'Fine.',
            future_field: 42
        });

        expect(result.success).toBe(true);
    });

    it('rejects a message whose type is not known', () => {
        expect(agentMessageSchema.safeParse({ type: 'invented' }).success).toBe(false);
    });
});

describe('compaction messages', () => {
    it('accepts estimated history counts and the run path', () => {
        const result = agentMessageSchema.safeParse({
            type: 'compaction',
            tokens_before: 21_300,
            tokens_after: 12_840,
            run_path: 'coder_streaming'
        });

        expect(result.success).toBe(true);
    });

    it('rejects a negative estimated token count', () => {
        expect(
            agentMessageSchema.safeParse({
                type: 'compaction',
                tokens_before: -1,
                tokens_after: 0,
                run_path: 'text_planning'
            }).success
        ).toBe(false);
    });
});

describe('task shapes', () => {
    const task = {
        id: 'abc',
        status: 'merged',
        prompt: 'Do the thing',
        repo_path: '/repo',
        branch_name: 'ai-task-abc',
        summary: 'Did the thing',
        created_at: '2026-01-01T00:00:00Z',
        updated_at: '2026-01-01T00:05:00Z'
    };

    it('accepts a task detail with its log', () => {
        const detail = parseResponse(
            taskDetailSchema,
            {
                task,
                coder_model: 'qwen',
                reviewer_model: 'llama3',
                log: [
                    {
                        entry_type: 'commit',
                        content: { message: 'work', hash: 'abcdef1234' },
                        created_at: '2026-01-01T00:01:00Z'
                    }
                ]
            },
            'task'
        );

        expect(detail.log).toHaveLength(1);
        expect(detail.task.status).toBe('merged');
    });

    it('defaults an absent log to empty rather than failing', () => {
        const detail = parseResponse(
            taskDetailSchema,
            { task, coder_model: 'qwen', reviewer_model: 'llama3' },
            'task'
        );

        expect(detail.log).toEqual([]);
    });

    it('rejects a task status the frontend does not know', () => {
        expect(() =>
            parseResponse(
                taskListSchema,
                { items: [{ ...task, status: 'exploded' }], total: 1, offset: 0, limit: 20 },
                'task list'
            )
        ).toThrow(SchemaValidationError);
    });

    it('rejects a directory listing that is not strings', () => {
        expect(() => parseResponse(directoryListingSchema, [1, 2], 'directory listing')).toThrow(
            SchemaValidationError
        );
    });
});
