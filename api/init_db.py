from api.database import get_connection


def init_db():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            prediction VARCHAR(10) NOT NULL,
            churn_probability FLOAT NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()

    cursor.close()
    connection.close()

    print("Prediction table created successfully.")


if __name__ == "__main__":
    init_db()