import { describe, expect, it } from 'vitest';
import { orderFilePaths, retainExpanded, toggleExpanded } from './diffSections';

describe('orderFilePaths', () => {
    it('orders paths alphabetically', () => {
        expect(orderFilePaths(['c.py', 'a.py', 'b.py'])).toEqual(['a.py', 'b.py', 'c.py']);
    });

    it('ignores case when ordering', () => {
        // A reader looking for README does not expect capitals to sort into
        // their own group ahead of everything else.
        expect(orderFilePaths(['zebra.py', 'README.md', 'apple.py'])).toEqual([
            'apple.py',
            'README.md',
            'zebra.py'
        ]);
    });

    it('orders by the full path, so directories group together', () => {
        expect(orderFilePaths(['src/z.py', 'docs/a.md', 'src/a.py', 'docs/z.md'])).toEqual([
            'docs/a.md',
            'docs/z.md',
            'src/a.py',
            'src/z.py'
        ]);
    });

    it('is total for paths differing only in case', () => {
        const ordered = orderFilePaths(['File.py', 'file.py']);
        expect(ordered).toHaveLength(2);
        // Same answer whatever order they arrived in.
        expect(orderFilePaths(['file.py', 'File.py'])).toEqual(ordered);
    });

    it('does not modify the array it was given', () => {
        const original = ['c.py', 'a.py'];
        orderFilePaths(original);
        expect(original).toEqual(['c.py', 'a.py']);
    });

    it('handles an empty list', () => {
        expect(orderFilePaths([])).toEqual([]);
    });
});

describe('retainExpanded', () => {
    it('keeps a file open when further diffs arrive', () => {
        const kept = retainExpanded(new Set(['a.py']), ['a.py', 'b.py']);

        expect(kept.has('a.py')).toBe(true);
    });

    it('leaves a newly arrived file collapsed', () => {
        const kept = retainExpanded(new Set(['a.py']), ['a.py', 'b.py']);

        expect(kept.has('b.py')).toBe(false);
    });

    it('drops a file that no longer has a diff', () => {
        const kept = retainExpanded(new Set(['a.py', 'gone.py']), ['a.py']);

        expect([...kept]).toEqual(['a.py']);
    });

    it('keeps several files open at once', () => {
        const kept = retainExpanded(new Set(['a.py', 'b.py']), ['a.py', 'b.py', 'c.py']);

        expect(kept.size).toBe(2);
    });

    it('does not modify the set it was given', () => {
        const original = new Set(['a.py', 'gone.py']);
        retainExpanded(original, ['a.py']);
        expect(original.size).toBe(2);
    });
});

describe('toggleExpanded', () => {
    it('opens a collapsed file', () => {
        expect(toggleExpanded(new Set(), 'a.py').has('a.py')).toBe(true);
    });

    it('closes an open file', () => {
        expect(toggleExpanded(new Set(['a.py']), 'a.py').has('a.py')).toBe(false);
    });

    it('leaves every other file alone', () => {
        const next = toggleExpanded(new Set(['a.py', 'b.py']), 'a.py');

        expect(next.has('b.py')).toBe(true);
        expect(next.has('a.py')).toBe(false);
    });

    it('does not modify the set it was given', () => {
        const original = new Set(['a.py']);
        toggleExpanded(original, 'b.py');
        expect(original.size).toBe(1);
    });
});
