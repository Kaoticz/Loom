import { browser } from '$app/environment';
import DOMPurify from 'dompurify';
import { marked } from 'marked';

marked.setOptions({ gfm: true, breaks: true });

/**
 * Escape HTML so text renders literally.
 *
 * Used as the server-side fallback, where DOMPurify has no DOM to work with.
 */
function escapeHtml(text: string): string {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

/**
 * Render markdown to HTML that is safe to insert into the page.
 *
 * Everything passed through here is model-influenced: agent reasoning, reviewer
 * comments, tool output. Markdown allows raw HTML through by default, so the
 * result is always sanitised before it reaches the DOM. Without that, anything
 * the model can be persuaded to emit could read the page's storage.
 */
export function renderMarkdown(text: string): string {
    const source = text ?? '';
    if (!browser) {
        return escapeHtml(source);
    }

    const html = marked.parse(source, { async: false }) as string;
    return DOMPurify.sanitize(html, {
        USE_PROFILES: { html: true },
        FORBID_TAGS: ['style', 'form', 'input', 'button'],
        FORBID_ATTR: ['style', 'srcset']
    });
}
