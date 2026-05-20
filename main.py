"""Entry point — starts the recommendation system API server."""

from src.api import create_app

if __name__ == "__main__":
    app = create_app()
    print("=" * 60)
    print("  Cross-Platform Recommendation System")
    print("  Server running at: http://localhost:5000")
    print()
    print("  Quick-start endpoints:")
    print("    GET  /health")
    print("    GET  /catalog")
    print("    GET  /recommend/content/<user_id>?n=10")
    print("    GET  /recommend/collaborative/<user_id>?n=10")
    print("    GET  /recommend/hybrid/<user_id>?n=10")
    print("    GET  /users/<user_id>/profile")
    print("    POST /events")
    print("=" * 60)
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1", host="0.0.0.0", port=5000)
