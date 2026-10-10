
from api.database import get_connection


def main():
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    prediction VARCHAR(10) NOT NULL,
                    churn_probability DOUBLE PRECISION NOT NULL,
                    model_version VARCHAR(50) NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """)

        connection.commit()
        print("Prediction table created successfully.")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
