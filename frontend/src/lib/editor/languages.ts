import type { Extension } from '@codemirror/state';

/**
 * Choosing syntax highlighting for a file in the diff viewer.
 *
 * The mapping from path to language is a pure function so it can be asserted
 * against every extension the project claims to support. Actually loading a
 * language pulls in a parser, so that is kept separate and deferred: only the
 * languages a diff actually contains are ever fetched.
 *
 * The set matches the languages `ast-grep` supports, so that a file Loom can
 * reason about structurally is also a file it can display properly.
 */

export type LanguageId =
    | 'bash'
    | 'c'
    | 'cpp'
    | 'csharp'
    | 'css'
    | 'elixir'
    | 'go'
    | 'haskell'
    | 'hcl'
    | 'html'
    | 'java'
    | 'javascript'
    | 'json'
    | 'kotlin'
    | 'lua'
    | 'nix'
    | 'php'
    | 'python'
    | 'ruby'
    | 'rust'
    | 'scala'
    | 'solidity'
    | 'swift'
    | 'typescript'
    | 'yaml';

/**
 * Which language each file extension indicates.
 *
 * Keys are lower case and carry no leading dot.
 */
const LANGUAGE_BY_EXTENSION: Readonly<Record<string, LanguageId>> = {
    // Web and frontend
    js: 'javascript',
    jsx: 'javascript',
    mjs: 'javascript',
    cjs: 'javascript',
    ts: 'typescript',
    cts: 'typescript',
    mts: 'typescript',
    tsx: 'typescript',
    html: 'html',
    htm: 'html',
    css: 'css',

    // Systems and low level
    c: 'c',
    h: 'c',
    cpp: 'cpp',
    cc: 'cpp',
    cxx: 'cpp',
    hpp: 'cpp',
    rs: 'rust',

    // General purpose and backend
    py: 'python',
    pyi: 'python',
    go: 'go',
    java: 'java',
    cs: 'csharp',
    php: 'php',
    rb: 'ruby',
    swift: 'swift',
    kt: 'kotlin',
    kts: 'kotlin',
    scala: 'scala',
    sbt: 'scala',

    // Scripting and functional
    sh: 'bash',
    zsh: 'bash',
    bash: 'bash',
    lua: 'lua',
    ex: 'elixir',
    exs: 'elixir',
    hs: 'haskell',
    nix: 'nix',

    // Config and data
    json: 'json',
    yml: 'yaml',
    yaml: 'yaml',
    hcl: 'hcl',
    sol: 'solidity'
};

/**
 * The extension of a file path, lower cased and without its dot.
 *
 * A leading dot marks a hidden file rather than an extension, so `.bashrc` has
 * no extension by this reckoning; only the part after a dot that follows at
 * least one other character counts.
 *
 * @param path The file path, which may include directories.
 * @returns The extension, or null when the file has none.
 */
export function extensionOf(path: string): string | null {
    const name = path.split(/[\\/]/).pop() ?? '';
    const dot = name.lastIndexOf('.');
    if (dot <= 0 || dot === name.length - 1) return null;
    return name.slice(dot + 1).toLowerCase();
}

/**
 * Which language a file should be highlighted as.
 *
 * @param path The file path.
 * @returns The language, or null when the file is shown as plain text.
 */
export function languageIdForPath(path: string): LanguageId | null {
    const extension = extensionOf(path);
    if (extension === null) return null;
    return LANGUAGE_BY_EXTENSION[extension] ?? null;
}

/** Every language the viewer can highlight. */
export function supportedLanguages(): LanguageId[] {
    return [...new Set(Object.values(LANGUAGE_BY_EXTENSION))].sort();
}

/** Every extension the viewer recognises. */
export function supportedExtensions(): string[] {
    return Object.keys(LANGUAGE_BY_EXTENSION).sort();
}

/** The stream-based legacy modes this viewer draws on. */
type LegacyMode = 'clike' | 'haskell' | 'lua' | 'ruby' | 'shell' | 'swift';

/** Explicit importers, one per legacy mode, so the bundler can find them. */
const LEGACY_MODES: Readonly<Record<LegacyMode, () => Promise<Record<string, unknown>>>> = {
    clike: () => import('@codemirror/legacy-modes/mode/clike'),
    haskell: () => import('@codemirror/legacy-modes/mode/haskell'),
    lua: () => import('@codemirror/legacy-modes/mode/lua'),
    ruby: () => import('@codemirror/legacy-modes/mode/ruby'),
    shell: () => import('@codemirror/legacy-modes/mode/shell'),
    swift: () => import('@codemirror/legacy-modes/mode/swift')
};

/**
 * Load the highlighting support for a language.
 *
 * Deferred rather than imported up front so that a diff of one Python file does
 * not carry the parsers for the other twenty-four languages with it.
 *
 * @param id The language to load.
 * @returns The editor extension, or null if the parser could not be loaded.
 */
export async function loadLanguage(id: LanguageId): Promise<Extension | null> {
    /**
     * Load one of the stream-based legacy modes.
     *
     * Each import is written out in full rather than built from a template,
     * because a template cannot be analysed at build time and the parser would
     * then be absent from the bundle and fail only in production.
     */
    const legacy = async (mode: LegacyMode, name: string): Promise<Extension> => {
        const { StreamLanguage } = await import('@codemirror/language');
        const parsers = await LEGACY_MODES[mode]();
        return StreamLanguage.define(parsers[name] as Parameters<typeof StreamLanguage.define>[0]);
    };

    switch (id) {
        case 'javascript':
            return (await import('@codemirror/lang-javascript')).javascript({ jsx: true });
        case 'typescript':
            return (await import('@codemirror/lang-javascript')).javascript({
                jsx: true,
                typescript: true
            });
        case 'html':
            return (await import('@codemirror/lang-html')).html();
        case 'css':
            return (await import('@codemirror/lang-css')).css();
        case 'c':
        case 'cpp':
            return (await import('@codemirror/lang-cpp')).cpp();
        case 'rust':
            return (await import('@codemirror/lang-rust')).rust();
        case 'python':
            return (await import('@codemirror/lang-python')).python();
        case 'go':
            return (await import('@codemirror/lang-go')).go();
        case 'java':
            return (await import('@codemirror/lang-java')).java();
        case 'php':
            return (await import('@codemirror/lang-php')).php();
        case 'json':
            return (await import('@codemirror/lang-json')).json();
        case 'yaml':
            return (await import('@codemirror/lang-yaml')).yaml();
        case 'elixir':
            return (await import('codemirror-lang-elixir')).elixir();
        case 'nix':
            return (await import('@replit/codemirror-lang-nix')).nix();
        case 'solidity':
            return (await import('@replit/codemirror-lang-solidity')).solidity;
        case 'hcl':
            return (await import('codemirror-lang-hcl')).hcl();
        case 'csharp':
            return legacy('clike', 'csharp');
        case 'kotlin':
            return legacy('clike', 'kotlin');
        case 'scala':
            return legacy('clike', 'scala');
        case 'ruby':
            return legacy('ruby', 'ruby');
        case 'swift':
            return legacy('swift', 'swift');
        case 'bash':
            return legacy('shell', 'shell');
        case 'lua':
            return legacy('lua', 'lua');
        case 'haskell':
            return legacy('haskell', 'haskell');
    }
}
