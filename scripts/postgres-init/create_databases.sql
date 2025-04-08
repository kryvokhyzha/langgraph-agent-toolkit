-- create_databases.sql
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'postgres') THEN
        CREATE DATABASE postgres;
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse') THEN
        CREATE DATABASE langfuse;
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'litellm') THEN
        CREATE DATABASE litellm;
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'agents') THEN
        CREATE DATABASE agents;
    END IF;
END
$$;
