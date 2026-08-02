import { describe, expect, it } from 'vitest';
import { loadLanguage, supportedLanguages } from './languages';

describe('language loaders', () => {
    for (const language of supportedLanguages()) {
        it(`loads a parser for ${language}`, async () => {
            const extension = await loadLanguage(language);
            expect(extension, `${language} resolved to nothing`).toBeTruthy();
        });
    }
});
