import js from '@eslint/js';
import ts from 'typescript-eslint';
import svelte from 'eslint-plugin-svelte';
import prettier from 'eslint-config-prettier';
import globals from 'globals';

/**
 * Lint rules for the frontend.
 *
 * `prettier` and `flat/prettier` come last so that formatting rules are switched
 * off here and left to Prettier alone, which is what stops the two tools from
 * disagreeing about the same line.
 */
export default ts.config(
    js.configs.recommended,
    ...ts.configs.recommended,
    ...svelte.configs['flat/recommended'],
    prettier,
    ...svelte.configs['flat/prettier'],
    {
        languageOptions: {
            globals: {
                ...globals.browser,
                ...globals.node
            }
        }
    },
    {
        rules: {
            // Rest destructuring is how a field is deliberately dropped, for
            // example to keep secrets out of persisted config.
            '@typescript-eslint/no-unused-vars': [
                'error',
                { ignoreRestSiblings: true, argsIgnorePattern: '^_' }
            ]
        }
    },
    {
        files: ['**/*.svelte'],
        languageOptions: {
            parserOptions: {
                parser: ts.parser
            }
        }
    },
    {
        ignores: ['node_modules/', 'build/', 'dist/', 'coverage/', '.svelte-kit/']
    }
);
