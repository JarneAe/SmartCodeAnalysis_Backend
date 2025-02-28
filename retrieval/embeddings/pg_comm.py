import psycopg
from dotenv import load_dotenv
import os

load_dotenv()

db_params = {
    "dbname": os.getenv("db_name"),
    "user": os.getenv("db_user"),
    "password": os.getenv("db_password"),
    "host": os.getenv("db_host"),
    "port": int(os.getenv("db_host_port"))
}

def get_codebase(codebase_id: str) -> {}:
    root_folder = get_root_folder_of_codebase(codebase_id)
    return get_children_of_folder(root_folder)

def get_root_folder_of_codebase(codebase_id: str) -> str:
    conn: psycopg.Connection
    with psycopg.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT "RootFolderId" FROM "Codebases" WHERE "Codebases"."Id" = %s
                """, (codebase_id,))

            return cur.fetchone()[0]


def get_children_of_folder(folder_id: str) -> {}:
    folder = {}

    conn: psycopg.Connection
    with psycopg.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                    SELECT * FROM "Files" 
                    WHERE ("Files"."Discriminator" LIKE 'CodeFile' OR "Files"."Discriminator" LIKE 'Folder')
                    AND "Files"."FolderId" = %s
                    """, (folder_id,))

            for row in cur.fetchall():
                if row[2] == "CodeFile":
                    folder[row[1]] = row[4]
                elif row[2] == "Folder":
                    folder[row[1]] = {"id": row[0]}

    for item in folder:
        if isinstance(folder[item], dict):
            folder[item] = get_children_of_folder(folder[item]["id"])

    return folder
