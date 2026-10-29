export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [2, 'always', ['docs', 'feat', 'fix', 'chore', 'refactor', 'perf', 'test', 'build', 'ci']],
    'scope-empty': [2, 'never'],          // type(scope): — scope is mandatory
    'subject-empty': [2, 'never'],
    'header-max-length': [2, 'always', 100],
  },
};
