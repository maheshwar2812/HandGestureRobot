import sqlite3
import hashlib


DATABASE_NAME = "users.db"


# =========================================================
# CREATE DATABASE
# =========================================================

def create_database():

    connection = sqlite3.connect(
        DATABASE_NAME
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            name TEXT NOT NULL,

            email TEXT UNIQUE NOT NULL,

            password TEXT NOT NULL

        )
        """
    )

    connection.commit()

    connection.close()


# =========================================================
# PASSWORD HASH
# =========================================================

def hash_password(password):

    return hashlib.sha256(
        password.encode()
    ).hexdigest()


# =========================================================
# CREATE USER
# =========================================================

def create_user(
    name,
    email,
    password
):

    # Make sure database exists
    create_database()

    connection = sqlite3.connect(
        DATABASE_NAME
    )

    cursor = connection.cursor()

    hashed_password = hash_password(
        password
    )

    try:

        cursor.execute(
            """
            INSERT INTO users
            (
                name,
                email,
                password
            )
            VALUES (?, ?, ?)
            """,
            (
                name,
                email,
                hashed_password
            )
        )

        connection.commit()

        return True, "Account created successfully!"

    except sqlite3.IntegrityError:

        return False, "Email already registered."

    finally:

        connection.close()


# =========================================================
# LOGIN USER
# =========================================================

def login_user(
    email,
    password
):

    # Make sure database exists
    create_database()

    connection = sqlite3.connect(
        DATABASE_NAME
    )

    cursor = connection.cursor()

    hashed_password = hash_password(
        password
    )

    cursor.execute(
        """
        SELECT
            id,
            name,
            email

        FROM users

        WHERE email = ?

        AND password = ?
        """,
        (
            email,
            hashed_password
        )
    )

    user = cursor.fetchone()

    connection.close()

    return user