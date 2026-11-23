"""
Database connection module
Supports MongoDB, PostgreSQL, and MySQL
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError
from datetime import datetime
import json

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def database_url():
    configured = os.getenv("DATABASE_URL")
    if configured:
        return configured
    import streamlit as st
    from streamlit.errors import StreamlitSecretNotFoundError

    try:
        configured = st.secrets.get("DATABASE_URL")
        if configured:
            return configured
    except StreamlitSecretNotFoundError:
        pass
    return URL.create("sqlite", database=str(ROOT / "mediai_demo.sqlite3"))

Base = declarative_base()


class User(Base):
    """User model for SQL databases"""
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)


class PredictionResult(Base):
    """Prediction result model for SQL databases"""
    __tablename__ = 'prediction_results'

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), index=True, nullable=False)
    disease_type = Column(String(50), nullable=False)  # 'lung', 'diabetes', 'kidney', 'breast_cancer'
    prediction_result = Column(String(100), nullable=False)  # 'Normal', 'Pneumonia', 'High Risk', etc.
    confidence = Column(Float)  # Confidence score or probability
    test_data = Column(Text)  # JSON string of input parameters
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class DatabaseConnection:
    """Unified database connection handler"""

    def __init__(self):
        self.db_url = database_url()
        if isinstance(self.db_url, str) and self.db_url.startswith(("mongodb://", "mongodb+srv://")):
            self.db_type = "mongodb"
        else:
            try:
                self.db_type = make_url(self.db_url).get_backend_name()
            except (ArgumentError, TypeError, ValueError):
                raise ValueError("Invalid database configuration.") from None
            if self.db_type not in {"sqlite", "postgresql", "mysql"}:
                raise ValueError("Unsupported database backend.")
        self.connection = None
        self.engine = None

    def connect(self):
        """Establish database connection"""
        if self.db_type == 'mongodb':
            return self._connect_mongodb()
        else:
            return self._connect_sql()

    def _connect_mongodb(self):
        """Connect to MongoDB"""
        client = MongoClient(self.db_url, serverSelectionTimeoutMS=5000)
        try:
            client.admin.command("ping")
            self.connection = client.get_default_database(default="medical_diagnosis_db")
        except Exception:
            client.close()
            raise
        return self.connection

    def _connect_sql(self):
        """Connect to SQL database (PostgreSQL/MySQL)"""
        self.engine = create_engine(self.db_url, hide_parameters=True)
        try:
            Base.metadata.create_all(self.engine)
            self.connection = sessionmaker(bind=self.engine)()
        except Exception:
            self.engine.dispose()
            raise
        return self.connection

    def get_connection(self):
        """Get or create connection"""
        if self.connection is None:
            self.connect()
        return self.connection

    def close(self):
        """Close database connection"""
        if self.connection is not None:
            if self.db_type == 'mongodb':
                self.connection.client.close()
            else:
                self.connection.close()
            self.connection = None
        if self.engine is not None:
            self.engine.dispose()
            self.engine = None


# MongoDB user operations
class MongoDBUserOps:
    """User operations for MongoDB"""

    def __init__(self, db):
        self.users = db['users']

    def create_user(self, username, email, password_hash, full_name=None):
        """Create a new user"""
        user_data = {
            'username': username,
            'email': email,
            'password_hash': password_hash,
            'full_name': full_name,
            'created_at': datetime.utcnow(),
            'last_login': None
        }
        result = self.users.insert_one(user_data)
        return result.inserted_id

    def get_user_by_username(self, username):
        """Get user by username"""
        return self.users.find_one({'username': username})

    def get_user_by_email(self, email):
        """Get user by email"""
        return self.users.find_one({'email': email})

    def update_last_login(self, username):
        """Update user's last login time"""
        self.users.update_one(
            {'username': username},
            {'$set': {'last_login': datetime.utcnow()}}
        )

    def user_exists(self, username=None, email=None):
        """Check if user exists"""
        query = {}
        if username:
            query['username'] = username
        if email:
            query['email'] = email
        return self.users.find_one(query) is not None


class MongoDBPredictionOps:
    """Prediction operations for MongoDB"""

    def __init__(self, db):
        self.predictions = db['prediction_results']

    def save_prediction(self, username, disease_type, prediction_result, confidence=None, test_data=None):
        """Save a prediction result"""
        prediction_data = {
            'username': username,
            'disease_type': disease_type,
            'prediction_result': prediction_result,
            'confidence': confidence,
            'test_data': test_data,
            'created_at': datetime.utcnow()
        }
        result = self.predictions.insert_one(prediction_data)
        return result.inserted_id

    def get_user_predictions(self, username, disease_type=None, limit=50):
        """Get user's prediction history"""
        query = {'username': username}
        if disease_type:
            query['disease_type'] = disease_type
        return list(self.predictions.find(query).sort('created_at', -1).limit(limit))

    def get_all_predictions(self, disease_type=None, limit=100):
        """Get all predictions (admin view)"""
        query = {}
        if disease_type:
            query['disease_type'] = disease_type
        return list(self.predictions.find(query).sort('created_at', -1).limit(limit))

    def count_user_predictions(self, username, disease_type=None):
        """Count user's predictions"""
        query = {'username': username}
        if disease_type:
            query['disease_type'] = disease_type
        return self.predictions.count_documents(query)


# SQL user operations
class SQLUserOps:
    """User operations for SQL databases"""

    def __init__(self, session):
        self.session = session

    def create_user(self, username, email, password_hash, full_name=None):
        """Create a new user"""
        user = User(
            username=username,
            email=email,
            password_hash=password_hash,
            full_name=full_name
        )
        self.session.add(user)
        self.session.commit()
        return user.id

    def get_user_by_username(self, username):
        """Get user by username"""
        return self.session.query(User).filter(User.username == username).first()

    def get_user_by_email(self, email):
        """Get user by email"""
        return self.session.query(User).filter(User.email == email).first()

    def update_last_login(self, username):
        """Update user's last login time"""
        user = self.get_user_by_username(username)
        if user:
            user.last_login = datetime.utcnow()
            self.session.commit()

    def user_exists(self, username=None, email=None):
        """Check if user exists"""
        query = self.session.query(User)
        if username:
            query = query.filter(User.username == username)
        if email:
            query = query.filter(User.email == email)
        return query.first() is not None


class SQLPredictionOps:
    """Prediction operations for SQL databases"""

    def __init__(self, session):
        self.session = session

    def save_prediction(self, username, disease_type, prediction_result, confidence=None, test_data=None):
        """Save a prediction result"""
        # Convert test_data dict to JSON string if provided
        test_data_json = json.dumps(test_data) if test_data else None

        prediction = PredictionResult(
            username=username,
            disease_type=disease_type,
            prediction_result=prediction_result,
            confidence=confidence,
            test_data=test_data_json
        )
        self.session.add(prediction)
        self.session.commit()
        return prediction.id

    def get_user_predictions(self, username, disease_type=None, limit=50):
        """Get user's prediction history"""
        query = self.session.query(PredictionResult).filter(
            PredictionResult.username == username
        )
        if disease_type:
            query = query.filter(PredictionResult.disease_type == disease_type)

        results = query.order_by(PredictionResult.created_at.desc()).limit(limit).all()

        # Convert to dict format for consistency with MongoDB
        return [{
            'id': r.id,
            'username': r.username,
            'disease_type': r.disease_type,
            'prediction_result': r.prediction_result,
            'confidence': r.confidence,
            'test_data': json.loads(r.test_data) if r.test_data else None,
            'created_at': r.created_at
        } for r in results]

    def get_all_predictions(self, disease_type=None, limit=100):
        """Get all predictions (admin view)"""
        query = self.session.query(PredictionResult)
        if disease_type:
            query = query.filter(PredictionResult.disease_type == disease_type)

        results = query.order_by(PredictionResult.created_at.desc()).limit(limit).all()

        return [{
            'id': r.id,
            'username': r.username,
            'disease_type': r.disease_type,
            'prediction_result': r.prediction_result,
            'confidence': r.confidence,
            'test_data': json.loads(r.test_data) if r.test_data else None,
            'created_at': r.created_at
        } for r in results]

    def count_user_predictions(self, username, disease_type=None):
        """Count user's predictions"""
        query = self.session.query(PredictionResult).filter(
            PredictionResult.username == username
        )
        if disease_type:
            query = query.filter(PredictionResult.disease_type == disease_type)
        return query.count()


# Factory function to get appropriate user operations
def get_user_operations():
    """Get user operations based on database type"""
    db_conn = DatabaseConnection()
    connection = db_conn.connect()

    if db_conn.db_type == 'mongodb':
        return MongoDBUserOps(connection), db_conn
    else:
        return SQLUserOps(connection), db_conn


def get_prediction_operations():
    """Get prediction operations based on database type"""
    db_conn = DatabaseConnection()
    connection = db_conn.connect()

    if db_conn.db_type == 'mongodb':
        return MongoDBPredictionOps(connection), db_conn
    else:
        return SQLPredictionOps(connection), db_conn
