import psycopg

DB_NAME = "voice_ai"
DB_USER = "vlad"
DB_PASSWORD = "Vn00193396"
DB_HOST = "10.2.4.87"
DB_PORT = "5445"

query = """
WITH linked_calls AS (
    SELECT DISTINCT
           linkedid,
           call_date
    FROM calls
    WHERE bid_id = %s
),
sorted_transcriptions AS (
    SELECT 
        t.linkedid,
        t.start,
        t.text,
        t.model,
        lc.call_date
    FROM transcribations t
    JOIN linked_calls lc ON t.linkedid = lc.linkedid
    WHERE t.text IS NOT NULL AND t.text <> ''
    ORDER BY lc.call_date, t.linkedid, t.start
)
SELECT linkedid, text, model
FROM sorted_transcriptions;
"""
conn = psycopg.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)
with conn.cursor() as cur:
    cur.execute(query, ('Пл2285564',))
    results = cur.fetchall()
    modeltext=set()

    conversations = {}
    for linkedid, text, model in results:
        if linkedid not in conversations:
            conversations[linkedid] = []
        conversations[linkedid].append(text)
        modeltext.add('linkedid ='+str(linkedid)+' model= '+str(model)) 

    full_text = []
    for linkedid, texts in conversations.items():
        conversation_text = ". ".join(text.strip() for text in texts)
        full_text.append(f"Следующий диалог в разговоре: {conversation_text}")

    final_text = "\n".join(full_text)

print(modeltext)