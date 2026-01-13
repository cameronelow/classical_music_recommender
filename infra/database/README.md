# Database Migrations

This directory contains SQL migrations for setting up and maintaining the Supabase database.

## Migrations

### 001_enable_rls_policies.sql

**Purpose:** Enable Row Level Security (RLS) on all tables to ensure data privacy and security.

**What it does:**
- Enables RLS on all application tables
- Creates policies so users can only access their own data
- Adds performance indexes for RLS policy checks
- Allows service role (backend) to log data for all users

**When to run:** Before deploying to production

**How to run:**
1. Go to your Supabase project dashboard
2. Navigate to SQL Editor
3. Copy and paste the contents of `001_enable_rls_policies.sql`
4. Click "Run" to execute the migration
5. Verify policies are created (see verification queries in the file)

## Database Schema

### Tables

#### recommendation_feedback
Stores user feedback (thumbs up/down) on recommendations.

**Columns:**
- `id` (uuid, primary key)
- `user_id` (text) - User identifier
- `work_id` (text) - Classical work identifier
- `rating` (integer) - -1 for thumbs down, 1 for thumbs up
- `comment` (text, optional) - User's comment
- `vibe` (text) - The search query that led to this recommendation
- `created_at` (timestamp)
- `updated_at` (timestamp)

**RLS Policy:** Users can only read/write their own feedback

#### saved_pieces
Stores user's saved/favorited classical pieces.

**Columns:**
- `id` (uuid, primary key)
- `user_id` (text) - User identifier
- `work_id` (text) - Classical work identifier
- `title` (text) - Work title
- `composer` (text) - Composer name
- `composer_id` (text, optional) - Composer identifier
- `vibe` (text, optional) - Original search query
- `explanation` (text, optional) - Why it matched
- `notes` (text, optional) - User's personal notes
- `saved_at` (timestamp)

**RLS Policy:** Users can only read/write their own saved pieces

#### recommendation_history
Audit log of all recommendations shown to users.

**Columns:**
- `id` (uuid, primary key)
- `user_id` (text) - User identifier
- `session_id` (text, optional) - Session identifier
- `work_id` (text) - Classical work identifier
- `composer_id` (text, optional) - Composer identifier
- `query` (text) - User's search query
- `vibe` (text) - Normalized vibe/mood
- `rank` (integer) - Position in results (1 = first)
- `relevance_score` (float) - Similarity score
- `created_at` (timestamp)

**RLS Policy:** Users can read their own history, service role can insert for all users

#### analytics_events
Application analytics and user behavior tracking.

**Columns:**
- `id` (uuid, primary key)
- `user_id` (text, nullable) - User identifier (null for anonymous)
- `session_id` (text, optional) - Session identifier
- `event_type` (text) - Type of event (e.g., 'search', 'play', 'share')
- `event_data` (jsonb) - Event-specific data
- `page_url` (text, optional) - Page URL
- `referrer` (text, optional) - Referrer URL
- `user_agent` (text, optional) - Browser user agent
- `created_at` (timestamp)

**RLS Policy:** Users can read their own events, anyone can insert (including anonymous)

## Security Best Practices

### 1. Service Key vs Anon Key

**Service Key (SUPABASE_SERVICE_KEY):**
- Bypasses ALL RLS policies
- Full database access
- **NEVER expose to frontend**
- Only use in backend API
- Store in backend/.env only

**Anon Key (SUPABASE_ANON_KEY):**
- Restricted by RLS policies
- Users can only access their own data
- Safe to expose in frontend
- Store in frontend/.env

### 2. Testing RLS Policies

Before deploying to production, test RLS policies:

```sql
-- Test as authenticated user
-- This should return only the user's own data
SELECT * FROM saved_pieces WHERE user_id = 'test-user-id';

-- Test cross-user access (should return empty)
SELECT * FROM saved_pieces WHERE user_id = 'different-user-id';
```

### 3. Monitoring

Set up monitoring for:
- Failed RLS policy checks (authorization errors)
- Unusual access patterns
- High volume of anonymous analytics events
- Service key usage (should only be backend)

## Adding New Migrations

When adding new migrations:

1. **Naming convention:** Use sequential numbers: `002_description.sql`, `003_description.sql`
2. **Include rollback:** Add commented rollback SQL at the bottom
3. **Test locally first:** Test in development before production
4. **Document changes:** Update this README with schema changes
5. **Version control:** Commit migrations to git (they don't contain secrets)

### Migration Template

```sql
-- Migration: [Number]_[description]
-- Description: [What this migration does]
-- Date: YYYY-MM-DD

-- Your migration SQL here

-- ============================================================================
-- Rollback (if needed)
-- ============================================================================

-- Rollback SQL here
```

## Troubleshooting

### RLS Policy Errors

**Error:** "new row violates row-level security policy"

**Solution:**
- Check that auth.uid() matches the user_id being inserted
- Verify user is authenticated (not anonymous)
- Check policy WITH CHECK clause

### Performance Issues

**Symptoms:** Slow queries after enabling RLS

**Solution:**
- Ensure indexes on user_id columns exist (migration includes these)
- Use EXPLAIN ANALYZE to check query plans
- Consider adding more specific indexes for common queries

### Anonymous Users

**Issue:** Analytics not working for anonymous users

**Solution:**
- Analytics events table allows NULL user_id
- Check frontend is passing session_id for anonymous tracking
- Policy allows INSERT with true (no authentication required)

## Maintenance

### Regular Tasks

**Weekly:**
- Review slow query logs
- Check for RLS policy violations in logs

**Monthly:**
- Clean up old analytics events (consider adding retention policy)
- Review and optimize indexes based on query patterns

**Quarterly:**
- Audit RLS policies for security
- Review database performance metrics
- Update indexes based on new query patterns

## Support

For database-related issues:
1. Check Supabase dashboard for error logs
2. Review RLS policies in SQL Editor
3. Test queries with EXPLAIN ANALYZE
4. Consult Supabase documentation: https://supabase.com/docs/guides/database
