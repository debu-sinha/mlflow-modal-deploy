# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.6.x   | :white_check_mark: |
| < 0.6   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in mlflow-modal-deploy, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email the maintainer directly or use GitHub's private vulnerability reporting feature
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Resolution target**: Within 30 days for critical issues

## Security Best Practices

When using mlflow-modal-deploy:

1. **Modal Authentication**: Keep your `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` secure. Never commit them to version control.

2. **Model Security**: Ensure MLflow models you deploy are from trusted sources. The plugin executes model code on Modal infrastructure.

3. **Network Security**: Modal endpoints are public by default. Consider implementing authentication in your model's predict function for sensitive use cases.

4. **Dependency Management**: Regularly update dependencies to patch known vulnerabilities.

## Dependency Security

This project uses:
- Dependabot for automated dependency updates
- CodeQL for static analysis
- Regular security audits of dependencies
