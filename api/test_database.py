from api.database import get_connection


def main():
    connection = get_connection()

    cursor = connection.cursor()
    cursor.execute("SELECT current_database(), current_user;")

    result = cursor.fetchone()

    print("===== POSTGRESQL CONNECTION =====")
    print(f"Database : {result[0]}")
    print(f"User     : {result[1]}")
    print("Status   : Connected successfully")

    cursor.close()
    connection.close()


if __name__ == "__main__":
    main()