-- Connect to agents database
\c agents;

-- Create schema for checkpoints
CREATE SCHEMA IF NOT EXISTS checkpoints;

-- Grant permissions to postgres user (not "agents" role which doesn't exist)
GRANT USAGE ON SCHEMA checkpoints TO postgres;
GRANT CREATE ON SCHEMA checkpoints TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA checkpoints TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA checkpoints TO postgres;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA checkpoints GRANT ALL ON TABLES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA checkpoints GRANT ALL ON SEQUENCES TO postgres;