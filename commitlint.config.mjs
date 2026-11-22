// Conventional Commits, enforced in CI. See CONTRIBUTING.md.
// Format:  type(scope): subject   e.g.  feat(rag): add hybrid retrieval
// ESM (.mjs) because the commitlint CI action loads config in an ES-module context.
export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'scope-enum': [
      2,
      'always',
      [
        'backend',
        'rag',
        'mcp',
        'models',
        'model-manager',
        'chat',
        'auth',
        'attachments',
        'frontend',
        'ui',
        'infra',
        'ci',
        'repo',
        'docs',
        'deps',
        'deps-dev',
      ],
    ],
    'subject-case': [2, 'never', ['upper-case', 'pascal-case', 'start-case']],
    // Detailed commit bodies are encouraged; don't fail on long lines/URLs.
    'body-max-line-length': [0, 'always', 100],
    'footer-max-line-length': [0, 'always', 100],
  },
};
