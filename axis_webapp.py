"""
aXis Vector Database Server
Simple Flask-based web interface for searching the aXis database
"""

from flask import Flask, render_template, request, jsonify
from axis_python.axis_db import aXisDB
import os
import logging
from pathlib import Path

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global database instance
current_db = None
current_db_path = None

# Default database
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "opmisizedst.db")


def get_available_databases():
    """Scan for .db files in the workspace."""
    db_dir = os.path.dirname(__file__)
    db_files = list(Path(db_dir).glob("*.db"))
    return [str(f.name) for f in db_files]


def load_database(db_name):
    """Load a database by name."""
    global current_db, current_db_path
    db_path = os.path.join(os.path.dirname(__file__), db_name)
    
    # Security check: ensure path is within the app directory
    if not os.path.abspath(db_path).startswith(os.path.abspath(os.path.dirname(__file__))):
        raise ValueError("Invalid database path")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_name}")
    
    current_db = aXisDB(db_path)
    current_db_path = db_path
    logger.info(f"Loaded database: {db_path}")


# Load default database on startup
load_database(os.path.basename(DEFAULT_DB_PATH))


@app.route("/")
def index():
    """Render the main landing page."""
    return render_template("index.html")


@app.route("/api/databases", methods=["GET"])
def get_databases():
    """Get list of available databases."""
    try:
        databases = get_available_databases()
        return jsonify({
            "databases": databases,
            "current": os.path.basename(current_db_path) if current_db_path else None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/load-database", methods=["POST"])
def load_db_endpoint():
    """Load a different database."""
    try:
        data = request.json
        db_name = data.get("database", "").strip()
        
        if not db_name:
            return jsonify({"error": "Database name required"}), 400
        
        load_database(db_name)
        return jsonify({
            "status": "success",
            "database": db_name,
            "vectors_loaded": len(current_db.vector_registry.vectors)
        })
    except Exception as e:
        logger.error(f"Load database error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["POST"])
def search():
    """Handle search requests and return results."""
    try:
        if not current_db:
            return jsonify({"error": "No database loaded"}), 500
        
        data = request.json
        query = data.get("query", "").strip()
        
        if not query:
            return jsonify({"error": "Empty query"}), 400
        
        # Search the database
        results = current_db.search(query, top_k=5)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "score": result.get("score", 0),
                "text": result.get("text", ""),
                "answer": result.get("answer", ""),
            })
        
        return jsonify({"results": formatted_results, "count": len(formatted_results)})
    
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/status", methods=["GET"])
def status():
    """Get database status."""
    try:
        if not current_db:
            return jsonify({"status": "error", "message": "No database loaded"}), 500
        
        vector_count = len(current_db.vector_registry.vectors)
        return jsonify({
            "status": "online",
            "database": os.path.basename(current_db_path) if current_db_path else "unknown",
            "vectors_loaded": vector_count
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/insert", methods=["POST"])
def insert():
    """Handle insert requests to add new data to the database."""
    try:
        if not current_db:
            return jsonify({"error": "No database loaded"}), 500
        
        data = request.json
        text = data.get("text", "").strip()
        payload = data.get("payload", {})
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
        
        # Insert into the database
        current_db.insert(text, payload)
        
        return jsonify({
            "status": "success",
            "message": "Data inserted successfully",
            "text": text,
            "payload": payload
        })
    
    except Exception as e:
        logger.error(f"Insert error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🚀 aXis Server starting...")
    print(f"📂 Database: {current_db_path}")
    print("🌐 Navigate to http://localhost:5005")
    app.run(debug=True, host="0.0.0.0", port=5005)