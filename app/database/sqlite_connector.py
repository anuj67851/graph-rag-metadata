import os
import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

class SQLiteConnector:
    """
    A connector for managing an SQLite database to store metadata about ingested files.
    This class provides a modular interface for file metadata operations.
    """
    _connection: Optional[sqlite3.Connection] = None

    def _get_connection(self) -> sqlite3.Connection:
        """Establishes and returns the SQLite database connection."""
        if self._connection is None:
            try:
                # 1. Get the directory path from the full database file path.
                db_dir = os.path.dirname(settings.SQLITE_DB_PATH)
                # 2. Create the directory if it doesn't already exist.
                os.makedirs(db_dir, exist_ok=True)

                self._connection = sqlite3.connect(settings.SQLITE_DB_PATH, check_same_thread=False)
                self._connection.row_factory = sqlite3.Row # Allows accessing columns by name
                logger.info(f"SQLite connection established to '{settings.SQLITE_DB_PATH}'.")
            except sqlite3.Error as e:
                logger.error(f"Error connecting to SQLite database: {e}", exc_info=True)
                raise
        return self._connection

    def close_connection(self):
        """Closes the SQLite database connection if it exists."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("SQLite connection closed.")

    def _execute_query(self, query: str, params: tuple = ()):
        """Executes a write query (INSERT, UPDATE, DELETE)."""
        conn = self._get_connection()
        try:
            with conn: # Using 'with' handles commit and rollback automatically
                conn.execute(query, params)
        except sqlite3.Error as e:
            logger.error(f"SQLite query failed: {query} with params {params}. Error: {e}", exc_info=True)
            raise

    def _fetch_one(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """Fetches a single record from the database."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None
        except sqlite3.Error as e:
            logger.error(f"SQLite fetch one failed: {query} with params {params}. Error: {e}", exc_info=True)
            return None

    def _fetch_all(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Fetches all records matching a query."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"SQLite fetch all failed: {query} with params {params}. Error: {e}", exc_info=True)
            return []

    def initialize_schema(self):
        """Creates the 'ingested_files' table if it doesn't exist."""
        query = """
                CREATE TABLE IF NOT EXISTS ingested_files (
                                                              filename TEXT PRIMARY KEY,
                                                              filepath TEXT NOT NULL,
                                                              filesize INTEGER NOT NULL,
                                                              ingestion_status TEXT NOT NULL,
                                                              ingested_at TIMESTAMP NOT NULL,
                                                              chunk_count INTEGER DEFAULT 0,
                                                              entities_added INTEGER DEFAULT 0,
                                                              relationships_added INTEGER DEFAULT 0,
                                                              error_message TEXT
                ); \
                """
        self._execute_query(query)
        logger.info("SQLite 'ingested_files' schema initialized.")

    def add_file_record(self, filename: str, filepath: str, filesize: int, status: str = "Pending") -> bool:
        """Adds a new file record to the database."""
        query = """
                INSERT INTO ingested_files (filename, filepath, filesize, ingestion_status, ingested_at)
                VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(filename) DO UPDATE SET
                    filepath=excluded.filepath,
                                                 filesize=excluded.filesize,
                                                 ingestion_status=excluded.ingestion_status,
                                                 ingested_at=excluded.ingested_at; \
                """
        params = (filename, filepath, filesize, status, datetime.utcnow())
        try:
            self._execute_query(query, params)
            logger.info(f"Added/Updated file record for '{filename}'.")
            return True
        except sqlite3.IntegrityError as e:
            logger.warning(f"Could not add record for '{filename}' due to integrity constraint: {e}")
            return False

    def update_file_status(self, filename: str, status: str, **kwargs):
        """
instaurated, 'error_message', 'chunk_count', 'entities_added', 'relationships_added').
        """
        fields_to_update = ["ingestion_status = ?"]
        params = [status]

        for key, value in kwargs.items():
            if key in ["chunk_count", "entities_added", "relationships_added", "error_message"]:
                fields_to_update.append(f"{key} = ?")
                params.append(value)

        # Always update the timestamp on status change
        fields_to_update.append("ingested_at = ?")
        params.append(datetime.utcnow())

        params.append(filename) # For the WHERE clause

        query = f"UPDATE ingested_files SET {', '.join(fields_to_update)} WHERE filename = ?"

        try:
            self._execute_query(query, tuple(params))
            logger.info(f"Updated status for '{filename}' to '{status}'.")
        except Exception as e:
            logger.error(f"Failed to update status for '{filename}': {e}", exc_info=True)

    def get_file_record(self, filename: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single file record by filename."""
        query = "SELECT * FROM ingested_files WHERE filename = ?"
        return self._fetch_one(query, (filename,))

    def list_all_files(self) -> List[Dict[str, Any]]:
        """Lists all file records in the database."""
        query = "SELECT * FROM ingested_files ORDER BY ingested_at DESC"
        return self._fetch_all(query)

    def delete_file_record(self, filename: str):
        """Deletes a file record from the database."""
        query = "DELETE FROM ingested_files WHERE filename = ?"
        self._execute_query(query, (filename,))
        logger.info(f"Deleted file record for '{filename}'.")

# --- Singleton Management for the Connector ---
_sqlite_connector_instance: Optional[SQLiteConnector] = None

def get_sqlite_connector() -> SQLiteConnector:
    """Provides a singleton instance of the SQLiteConnector."""
    global _sqlite_connector_instance
    if _sqlite_connector_instance is None:
        _sqlite_connector_instance = SQLiteConnector()
    return _sqlite_connector_instance