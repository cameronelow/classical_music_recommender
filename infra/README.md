# Infrastructure & Deployment Tools

This directory contains scripts and configurations for deploying and securing the Classical Music Recommender.

## 📁 Directory Structure

```
infra/
├── README.md                   # This file
├── setup_production.sh         # Interactive production setup script
├── test_security.sh            # Security testing suite
└── database/
    ├── README.md               # Database documentation
    └── migrations/
        └── 001_enable_rls_policies.sql  # RLS security migration
```

## 🚀 Quick Start

### 1. Set Up Production Environment

Run the interactive setup script:

```bash
./infra/setup_production.sh
```

This will:
- Create `.env` files from templates
- Guide you through configuration
- Verify git exclusions
- Provide deployment checklist

### 2. Apply Database Security

Run the RLS migration in Supabase:

1. Go to [Supabase Dashboard](https://supabase.com/dashboard) → SQL Editor
2. Copy contents of `infra/database/migrations/001_enable_rls_policies.sql`
3. Execute the migration
4. Verify policies with verification queries in the file

See `database/README.md` for detailed documentation.

### 3. Test Security Features

Run the security test suite:

```bash
./infra/test_security.sh
```

## 📋 Available Scripts

### setup_production.sh

Interactive script to set up production environment variables.

**Usage:**
```bash
./infra/setup_production.sh
```

### test_security.sh

Automated security testing suite.

**Usage:**
```bash
# Test local
./infra/test_security.sh

# Test production
API_URL=https://api.yourdomain.com ./infra/test_security.sh
```

**Tests:** Security headers, rate limiting, CORS, input validation, monitoring

## 📚 Documentation

- [SECURITY.md](../SECURITY.md) - Complete security guide
- [DEPLOYMENT.md](../DEPLOYMENT.md) - Deployment instructions
- [database/README.md](database/README.md) - Database documentation
- [backend/security_monitor.py](../backend/security_monitor.py) - Security monitoring

## 🔒 Security Features

- **Rate Limiting:** Global (60/min) + search endpoints (20/min)
- **Security Headers:** CSP, HSTS, X-Frame-Options, etc.
- **CORS Protection:** Configurable allowed origins
- **Security Monitoring:** Automated logging and alerts
- **Row Level Security:** Supabase RLS policies
- **Input Validation:** Pydantic models

See SECURITY.md for complete details.
