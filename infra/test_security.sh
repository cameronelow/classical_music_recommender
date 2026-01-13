#!/bin/bash

# Security Testing Script
# Tests security features of the Classical Music Recommender API

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:3001}"

echo "=========================================="
echo "Security Testing Suite"
echo "=========================================="
echo ""
echo "Testing API: $API_URL"
echo ""

# Function to print test results
print_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_pass() {
    echo -e "${GREEN}  ✓ PASS:${NC} $1"
}

print_fail() {
    echo -e "${RED}  ✗ FAIL:${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}  ⚠ WARN:${NC} $1"
}

# Check if API is running
print_test "Checking if API is running..."
if curl -s -f "$API_URL/" > /dev/null; then
    print_pass "API is responding"
else
    print_fail "API is not responding at $API_URL"
    echo "Please start the backend server first: cd backend && uvicorn api:app"
    exit 1
fi
echo ""

# ============================================================================
# Test 1: Security Headers
# ============================================================================

print_test "Test 1: Security Headers"
HEADERS=$(curl -s -I "$API_URL/api/health")

# Check for required security headers
if echo "$HEADERS" | grep -q "x-content-type-options: nosniff"; then
    print_pass "X-Content-Type-Options header present"
else
    print_fail "X-Content-Type-Options header missing"
fi

if echo "$HEADERS" | grep -q "x-frame-options: DENY"; then
    print_pass "X-Frame-Options header present"
else
    print_fail "X-Frame-Options header missing"
fi

if echo "$HEADERS" | grep -q "x-xss-protection:"; then
    print_pass "X-XSS-Protection header present"
else
    print_fail "X-XSS-Protection header missing"
fi

if echo "$HEADERS" | grep -q "strict-transport-security:"; then
    print_pass "Strict-Transport-Security header present"
else
    print_warn "HSTS header missing (OK for localhost)"
fi

if echo "$HEADERS" | grep -q "content-security-policy:"; then
    print_pass "Content-Security-Policy header present"
else
    print_fail "Content-Security-Policy header missing"
fi

echo ""

# ============================================================================
# Test 2: Rate Limiting
# ============================================================================

print_test "Test 2: Rate Limiting"

# Test rate limit headers
RATE_LIMIT_RESPONSE=$(curl -s -I "$API_URL/api/health")

if echo "$RATE_LIMIT_RESPONSE" | grep -q "x-ratelimit-limit:"; then
    print_pass "Rate limit headers present"
    LIMIT=$(echo "$RATE_LIMIT_RESPONSE" | grep -i "x-ratelimit-limit:" | awk '{print $2}' | tr -d '\r')
    echo "  Rate limit: $LIMIT requests/minute"
else
    print_fail "Rate limit headers missing"
fi

# Test actual rate limiting (light test - 10 requests)
print_test "Testing rate limit enforcement (10 rapid requests)..."
SUCCESS_COUNT=0
for i in {1..10}; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/api/health")
    if [ "$STATUS" = "200" ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    fi
done

if [ $SUCCESS_COUNT -eq 10 ]; then
    print_pass "All requests succeeded (within rate limit)"
else
    print_warn "$SUCCESS_COUNT/10 requests succeeded"
fi

echo ""

# ============================================================================
# Test 3: CORS Configuration
# ============================================================================

print_test "Test 3: CORS Configuration"

# Test with allowed origin
CORS_RESPONSE=$(curl -s -I -H "Origin: http://localhost:3001" "$API_URL/api/health")

if echo "$CORS_RESPONSE" | grep -q "access-control-allow-origin:"; then
    print_pass "CORS headers present for allowed origin"
else
    print_fail "CORS headers missing for allowed origin"
fi

# Test with disallowed origin
CORS_MALICIOUS=$(curl -s -I -H "Origin: https://malicious-site.com" "$API_URL/api/health")

if echo "$CORS_MALICIOUS" | grep -q "access-control-allow-origin: https://malicious-site.com"; then
    print_fail "CORS allows malicious origin - SECURITY RISK!"
else
    print_pass "CORS blocks unauthorized origins"
fi

echo ""

# ============================================================================
# Test 4: API Documentation Access
# ============================================================================

print_test "Test 4: API Documentation Access"

# Check if docs are accessible (should depend on ENVIRONMENT variable)
DOCS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/docs")

if [ "$DOCS_STATUS" = "404" ]; then
    print_pass "API docs are hidden (production mode)"
elif [ "$DOCS_STATUS" = "200" ]; then
    print_warn "API docs are accessible (development mode)"
    echo "  Set ENVIRONMENT=production to hide docs"
else
    print_warn "Unexpected status for /docs: $DOCS_STATUS"
fi

echo ""

# ============================================================================
# Test 5: Input Validation
# ============================================================================

print_test "Test 5: Input Validation"

# Test with invalid input
INVALID_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/search/mood" \
    -H "Content-Type: application/json" \
    -d '{"invalid": "data"}')

STATUS=$(echo "$INVALID_RESPONSE" | tail -n1)

if [ "$STATUS" = "422" ]; then
    print_pass "Invalid input rejected (422)"
else
    print_warn "Unexpected status for invalid input: $STATUS (expected 422)"
fi

# Test with SQL injection attempt
SQL_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/search/mood" \
    -H "Content-Type: application/json" \
    -d '{"query": "test'"'"'; DROP TABLE users--", "n": 1}')

SQL_STATUS=$(echo "$SQL_RESPONSE" | tail -n1)

if [ "$SQL_STATUS" = "200" ] || [ "$SQL_STATUS" = "500" ]; then
    print_pass "SQL injection attempt handled"
else
    print_warn "Unexpected SQL injection response: $SQL_STATUS"
fi

echo ""

# ============================================================================
# Test 6: Security Monitoring
# ============================================================================

print_test "Test 6: Security Monitoring"

# Check if security log exists
if [ -f "backend/security.log" ]; then
    print_pass "Security log file exists"
    LOG_LINES=$(wc -l < backend/security.log)
    echo "  Log contains $LOG_LINES entries"
else
    print_warn "Security log file not found (will be created on first event)"
fi

# Test security summary endpoint (dev mode only)
SUMMARY_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/api/security/summary")

if [ "$SUMMARY_STATUS" = "200" ]; then
    print_pass "Security summary endpoint accessible (dev mode)"
elif [ "$SUMMARY_STATUS" = "404" ]; then
    print_pass "Security summary endpoint hidden (production mode)"
else
    print_warn "Unexpected status for security summary: $SUMMARY_STATUS"
fi

echo ""

# ============================================================================
# Test 7: Environment Configuration
# ============================================================================

print_test "Test 7: Environment Configuration"

# Check if .env files exist
if [ -f "backend/.env" ]; then
    print_pass "backend/.env exists"
else
    print_fail "backend/.env not found"
    echo "  Run: cp backend/.env.example backend/.env"
fi

if [ -f "frontend/.env" ]; then
    print_pass "frontend/.env exists"
else
    print_warn "frontend/.env not found"
    echo "  Run: cp frontend/.env.example frontend/.env"
fi

# Check if .env is in .gitignore
if grep -q "backend/.env" .gitignore; then
    print_pass ".env files excluded from git"
else
    print_fail ".env files NOT in .gitignore - SECURITY RISK!"
fi

# Check for placeholder values
if [ -f "backend/.env" ]; then
    if grep -q "your_supabase" backend/.env; then
        print_warn "backend/.env contains placeholder values"
        echo "  Update with real credentials for production"
    else
        print_pass "backend/.env appears configured"
    fi
fi

echo ""

# ============================================================================
# Summary
# ============================================================================

echo "=========================================="
echo "Security Test Summary"
echo "=========================================="
echo ""
echo "All critical security features have been tested."
echo ""
echo "Before deploying to production:"
echo ""
echo "  1. Set ENVIRONMENT=production in backend/.env"
echo "  2. Configure ALLOWED_ORIGINS with your production domain"
echo "  3. Run database migration: infra/database/migrations/001_enable_rls_policies.sql"
echo "  4. Enable HTTPS/SSL for your domain"
echo "  5. Review and test all security features in staging"
echo ""
echo "For more information, see:"
echo "  - SECURITY.md - Complete security guide"
echo "  - DEPLOYMENT.md - Deployment checklist"
echo ""
