-- Create required databases - using simple statements (not in functions/blocks)
CREATE DATABASE langfuse;
CREATE DATABASE litellm;
CREATE DATABASE agents;

-- Grant all privileges to postgres user
ALTER DATABASE langfuse OWNER TO postgres;
ALTER DATABASE litellm OWNER TO postgres;
ALTER DATABASE agents OWNER TO postgres;
