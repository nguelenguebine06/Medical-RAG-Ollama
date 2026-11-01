from pathlib import Path
from flask import Flask, send_from_directory
from flask_cors import CORS

STATIC_DIR = Path(__file__).resolve().parent / "src" / "api" / "static"

def create_app():
    app = Flask(__name__, static_folder=str(STATIC_DIR))
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Blueprints
    from src.api.routes.chat import chat_bp
    from src.api.routes.ingestion import ingestion_bp
    app.register_blueprint(chat_bp, url_prefix='/api')
    app.register_blueprint(ingestion_bp, url_prefix='/api')

    # Route principale
    @app.route('/')
    def index():
        return send_from_directory(STATIC_DIR, 'index.html')

    # Gestionnaires d'erreurs
    @app.errorhandler(404)
    def not_found(e):
        return {'error': 'Route non trouvée'}, 404

    @app.errorhandler(500)
    def server_error(e):
        return {'error': 'Erreur serveur'}, 500

    return app

app = create_app()

if __name__ == "__main__":
    app.run(port=5000, debug=True)