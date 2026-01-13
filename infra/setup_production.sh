#!/bin/bash

# Production Environment Setup Script
# This script helps set up production environment variables securely

set -e  # Exit on error

echo "=========================================="
echo "Classical Music Recommender"
echo "Production Environment Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${NC}ℹ $1${NC}"
}

# Check if running in project root
if [ ! -f "setup.py" ] && [ ! -f "package.json" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_info "This script will help you set up environment variables for production"
echo ""

# ============================================================================
# Backend Setup
# ============================================================================

echo "=========================================="
echo "Backend Configuration"
echo "=========================================="
echo ""

if [ -f "backend/.env" ]; then
    print_warning "backend/.env already exists"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping backend setup"
    else
        rm backend/.env
    fi
fi

if [ ! -f "backend/.env" ]; then
    print_info "Creating backend/.env from template..."
    cp backend/.env.example backend/.env
    print_success "Created backend/.env"
    echo ""

    print_warning "IMPORTANT: You need to fill in the following values in backend/.env:"
    echo ""
    echo "  1. SUPABASE_URL - Get from Supabase Dashboard → Settings → API"
    echo "  2. SUPABASE_SERVICE_KEY - Get from Supabase Dashboard → Settings → API"
    echo "  3. BACKEND_URL - Your production backend domain (e.g., https://api.yourdomain.com)"
    echo "  4. FRONTEND_URL - Your production frontend domain (e.g., https://app.yourdomain.com)"
    echo "  5. ALLOWED_ORIGINS - Your production frontend domain (for CORS)"
    echo "  6. ENVIRONMENT=production - Set this to disable API docs"
    echo ""

    read -p "Do you want to open backend/.env in your editor now? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        if command -v code &> /dev/null; then
            code backend/.env
        elif command -v nano &> /dev/null; then
            nano backend/.env
        elif command -v vi &> /dev/null; then
            vi backend/.env
        else
            print_info "Please edit backend/.env manually"
        fi
    fi
    echo ""
fi

# ============================================================================
# Frontend Setup
# ============================================================================

echo "=========================================="
echo "Frontend Configuration"
echo "=========================================="
echo ""

if [ -f "frontend/.env" ]; then
    print_warning "frontend/.env already exists"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping frontend setup"
    else
        rm frontend/.env
    fi
fi

if [ ! -f "frontend/.env" ]; then
    print_info "Creating frontend/.env from template..."
    cp frontend/.env.example frontend/.env
    print_success "Created frontend/.env"
    echo ""

    print_warning "IMPORTANT: You need to fill in the following values in frontend/.env:"
    echo ""
    echo "  1. NEXT_PUBLIC_API_URL - Your production backend URL"
    echo "  2. NEXT_PUBLIC_SUPABASE_URL - Same as backend SUPABASE_URL"
    echo "  3. NEXT_PUBLIC_SUPABASE_ANON_KEY - Get from Supabase Dashboard (NOT service key!)"
    echo ""

    read -p "Do you want to open frontend/.env in your editor now? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        if command -v code &> /dev/null; then
            code frontend/.env
        elif command -v nano &> /dev/null; then
            nano frontend/.env
        elif command -v vi &> /dev/null; then
            vi frontend/.env
        else
            print_info "Please edit frontend/.env manually"
        fi
    fi
    echo ""
fi

# ============================================================================
# Verification
# ============================================================================

echo "=========================================="
echo "Verification Checks"
echo "=========================================="
echo ""

# Check if .env files exist
if [ -f "backend/.env" ]; then
    print_success "backend/.env exists"
else
    print_error "backend/.env not found"
fi

if [ -f "frontend/.env" ]; then
    print_success "frontend/.env exists"
else
    print_error "frontend/.env not found"
fi

# Check if .env files are in .gitignore
if grep -q "backend/.env" .gitignore; then
    print_success ".env files are excluded from git"
else
    print_error ".env files are NOT excluded from git - SECURITY RISK!"
fi

# Check for placeholder values in backend/.env
if [ -f "backend/.env" ]; then
    if grep -q "your_supabase" backend/.env; then
        print_warning "backend/.env still contains placeholder values"
        print_info "Update backend/.env with your actual credentials"
    else
        print_success "backend/.env appears configured"
    fi
fi

# Check for placeholder values in frontend/.env
if [ -f "frontend/.env" ]; then
    if grep -q "your_supabase" frontend/.env; then
        print_warning "frontend/.env still contains placeholder values"
        print_info "Update frontend/.env with your actual credentials"
    else
        print_success "frontend/.env appears configured"
    fi
fi

echo ""

# ============================================================================
# Security Checklist
# ============================================================================

echo "=========================================="
echo "Security Checklist"
echo "=========================================="
echo ""

print_info "Before deploying to production, ensure:"
echo ""
echo "  [ ] backend/.env has ENVIRONMENT=production"
echo "  [ ] ALLOWED_ORIGINS only includes your production domain"
echo "  [ ] SUPABASE_SERVICE_KEY is only in backend/.env (NOT frontend)"
echo "  [ ] frontend/.env uses SUPABASE_ANON_KEY (NOT service key)"
echo "  [ ] All .env files are excluded from git"
echo "  [ ] You've run the database migration: infra/database/migrations/001_enable_rls_policies.sql"
echo "  [ ] SSL/HTTPS is configured for your domain"
echo "  [ ] Rate limits are appropriate for your expected traffic"
echo "  [ ] You've tested the app in staging environment"
echo ""

# ============================================================================
# Next Steps
# ============================================================================

echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""

print_info "1. Database Setup"
echo "   - Go to Supabase Dashboard → SQL Editor"
echo "   - Run: infra/database/migrations/001_enable_rls_policies.sql"
echo ""

print_info "2. Test Locally"
echo "   - Start backend: cd backend && uvicorn api:app --reload"
echo "   - Start frontend: cd frontend && npm run dev"
echo "   - Test all features work with production .env files"
echo ""

print_info "3. Deploy"
echo "   - Follow the deployment guide: DEPLOYMENT.md"
echo "   - Set environment variables in your hosting platform"
echo "   - Verify security headers with: curl -I https://your-api-url.com"
echo ""

print_info "4. Monitor"
echo "   - Check backend logs for rate limit violations"
echo "   - Monitor Supabase dashboard for RLS policy errors"
echo "   - Set up alerts for 5xx errors"
echo ""

print_success "Setup complete!"
echo ""
print_info "For more information, see:"
echo "  - SECURITY.md - Complete security documentation"
echo "  - DEPLOYMENT.md - Deployment guide"
echo "  - infra/database/README.md - Database setup"
echo ""
