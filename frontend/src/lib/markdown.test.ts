import { describe, expect, it } from 'vitest';

import { renderMarkdown } from '$lib/markdown';

/**
 * The sanitiser between model output and the page.
 *
 * Everything rendered through here is model-influenced — agent reasoning,
 * reviewer comments, tool output — and it reaches the DOM via `{@html}`. The
 * sanitising was correct when this was written and nothing asserted it, so
 * relaxing an option later would have opened the page to script injection with
 * every test still green. These assert the security property directly rather
 * than the configuration that currently provides it, so the tests survive a
 * change of library.
 */
describe('rendering model output as markdown', () => {
    it('renders ordinary markdown', () => {
        const html = renderMarkdown('# Heading\n\nSome **bold** text.');

        expect(html).toContain('<h1');
        expect(html).toContain('<strong>bold</strong>');
    });

    it('strips a script tag while keeping the text around it', () => {
        const html = renderMarkdown('before<script>alert(1)</script>after');

        expect(html).not.toContain('<script');
        expect(html).not.toContain('alert(1)');
        expect(html).toContain('before');
        expect(html).toContain('after');
    });

    it('strips an inline event handler but keeps the element', () => {
        // The attribute is the danger, not the tag, and removing the whole
        // element would silently lose content the reader was meant to see.
        const html = renderMarkdown('<img src="x" onerror="alert(1)">');

        expect(html).not.toContain('onerror');
        expect(html).not.toContain('alert(1)');
        expect(html).toContain('<img');
    });

    it('removes a javascript: link target', () => {
        const html = renderMarkdown('[click me](javascript:alert(1))');

        expect(html.toLowerCase()).not.toContain('javascript:');
        // The text survives, so the reader still sees what was written.
        expect(html).toContain('click me');
    });

    it('strips an iframe outright', () => {
        const html = renderMarkdown('<iframe src="https://example.com"></iframe>');

        expect(html).not.toContain('<iframe');
    });

    it('strips the tags the renderer forbids by name', () => {
        // These are forbidden explicitly rather than by profile, so a change to
        // that list is worth catching.
        for (const markup of [
            '<style>body{display:none}</style>',
            '<form action="/x"><input name="a"></form>',
            '<button onclick="alert(1)">go</button>'
        ]) {
            const html = renderMarkdown(markup);

            expect(html).not.toContain('<style');
            expect(html).not.toContain('<form');
            expect(html).not.toContain('<input');
            expect(html).not.toContain('<button');
        }
    });

    it('strips a style attribute, which can hide or reposition content', () => {
        const html = renderMarkdown('<p style="position:fixed;top:0">covering</p>');

        expect(html).not.toContain('style=');
        expect(html).toContain('covering');
    });

    it('keeps code blocks intact, escaping what is inside them', () => {
        // Agent output is full of code, and a sanitiser that mangled it would
        // be noticed immediately — but a sanitiser that executed it would not.
        const html = renderMarkdown('```html\n<script>alert(1)</script>\n```');

        expect(html).toContain('<code');
        expect(html).not.toContain('<script>alert');
        expect(html).toContain('alert(1)');
    });

    it('handles empty and whitespace input without failing', () => {
        expect(renderMarkdown('')).toBe('');
        expect(() => renderMarkdown('   \n\n  ')).not.toThrow();
    });

    it('handles malformed markup rather than throwing', () => {
        // Model output is not guaranteed to be well formed, and a render that
        // threw would take the whole message list down with it.
        for (const malformed of ['<div><span>unclosed', '<<>>', '](broken](', '<a href=']) {
            expect(() => renderMarkdown(malformed)).not.toThrow();
        }
    });
});
