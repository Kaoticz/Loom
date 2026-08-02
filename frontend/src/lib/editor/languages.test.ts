import { describe, expect, it } from 'vitest';
import {
    extensionOf,
    languageIdForPath,
    supportedExtensions,
    supportedLanguages,
    type LanguageId
} from './languages';

/**
 * Every extension the project claims to support, grouped as the requirement
 * states them. Asserting against this list rather than against the mapping's own
 * keys means a language quietly dropped from the mapping fails here.
 */
const EXPECTED: Readonly<Record<LanguageId, string[]>> = {
    javascript: ['js', 'jsx', 'mjs', 'cjs'],
    typescript: ['ts', 'cts', 'mts', 'tsx'],
    html: ['html', 'htm'],
    css: ['css'],
    c: ['c', 'h'],
    cpp: ['cpp', 'cc', 'cxx', 'hpp'],
    rust: ['rs'],
    python: ['py', 'pyi'],
    go: ['go'],
    java: ['java'],
    csharp: ['cs'],
    php: ['php'],
    ruby: ['rb'],
    swift: ['swift'],
    kotlin: ['kt', 'kts'],
    scala: ['scala', 'sbt'],
    bash: ['sh', 'zsh', 'bash'],
    lua: ['lua'],
    elixir: ['ex', 'exs'],
    haskell: ['hs'],
    nix: ['nix'],
    json: ['json'],
    yaml: ['yml', 'yaml'],
    hcl: ['hcl'],
    solidity: ['sol']
};

describe('extensionOf', () => {
    it('reads the extension from a bare filename', () => {
        expect(extensionOf('main.py')).toBe('py');
    });

    it('reads it from a nested path', () => {
        expect(extensionOf('src/lib/stores/agent.ts')).toBe('ts');
    });

    it('handles Windows separators', () => {
        expect(extensionOf('src\\lib\\agent.ts')).toBe('ts');
    });

    it('takes the last extension of several', () => {
        expect(extensionOf('archive.tar.gz')).toBe('gz');
    });

    it('lower cases it', () => {
        expect(extensionOf('README.MD')).toBe('md');
        expect(extensionOf('Main.PY')).toBe('py');
    });

    it('treats a hidden file as having no extension', () => {
        // `.bashrc` is a hidden file, not a `bashrc` file.
        expect(extensionOf('.bashrc')).toBeNull();
        expect(extensionOf('/home/user/.gitignore')).toBeNull();
    });

    it('returns null when there is no extension', () => {
        expect(extensionOf('Makefile')).toBeNull();
        expect(extensionOf('src/bin/loom')).toBeNull();
    });

    it('returns null for a trailing dot', () => {
        expect(extensionOf('weird.')).toBeNull();
    });

    it('returns null for an empty path', () => {
        expect(extensionOf('')).toBeNull();
    });
});

describe('languageIdForPath', () => {
    for (const [language, extensions] of Object.entries(EXPECTED) as [LanguageId, string[]][]) {
        for (const extension of extensions) {
            it(`maps .${extension} to ${language}`, () => {
                expect(languageIdForPath(`some/dir/file.${extension}`)).toBe(language);
            });
        }
    }

    it('is case insensitive about the extension', () => {
        expect(languageIdForPath('Program.CS')).toBe('csharp');
        expect(languageIdForPath('Build.SBT')).toBe('scala');
    });

    it('falls back to no highlighting for an unknown extension', () => {
        expect(languageIdForPath('notes.md')).toBeNull();
        expect(languageIdForPath('image.png')).toBeNull();
        expect(languageIdForPath('component.svelte')).toBeNull();
    });

    it('falls back to no highlighting for an extensionless file', () => {
        expect(languageIdForPath('Dockerfile')).toBeNull();
        expect(languageIdForPath('LICENSE')).toBeNull();
    });
});

describe('supported set', () => {
    it('covers exactly the languages the project claims', () => {
        expect(supportedLanguages()).toEqual(Object.keys(EXPECTED).sort());
    });

    it('covers exactly the extensions the project claims', () => {
        const claimed = Object.values(EXPECTED).flat().sort();
        expect(supportedExtensions()).toEqual(claimed);
    });

    it('maps each extension to only one language', () => {
        const seen = new Map<string, LanguageId>();
        for (const [language, extensions] of Object.entries(EXPECTED) as [LanguageId, string[]][]) {
            for (const extension of extensions) {
                expect(seen.has(extension), `${extension} claimed twice`).toBe(false);
                seen.set(extension, language);
            }
        }
    });
});
