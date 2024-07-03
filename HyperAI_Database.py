import random
import sqlite3 as sl

sl.threadsafety = 3
from datetime import datetime
from typing import Union
import traceback
import time


def eztime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def tm(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


def db_init(db, cursor):  # TEXT as ISO8601 strings ("YYYY-MM-DD HH:MM:SS.SSS").
    # первая таблица общая
    cursor.execute("""
CREATE TABLE users (
    user_id            INTEGER PRIMARY KEY AUTOINCREMENT
                               UNIQUE
                               NOT NULL,
    name               TEXT,
    rank               REAL,
    firstreg           TEXT    NOT NULL,
    last_interact      TEXT    NOT NULL,
    last_answered      TEXT,
    diags_count        INTEGER DEFAULT (0) 
                               NOT NULL,
    last_question_date TEXT,
    last_question      TEXT
);



""")
    # вторая таблица с никами
    cursor.execute("""
CREATE TABLE users_nicknames (
    nick_id    INTEGER PRIMARY KEY AUTOINCREMENT
                       NOT NULL,
    user_id    NUMERIC REFERENCES users (user_id) ON DELETE CASCADE
                       NOT NULL,
    nick       TEXT    NOT NULL,
    env        TEXT,
    server     TEXT,
    other_info TEXT,
    added_date TEXT    NOT NULL,
    UNIQUE (
        nick,
        env
    )
);

""")
    # третья таблица с диалогами
    cursor.execute("""
CREATE TABLE users_dialogs (
    diag_id         INTEGER PRIMARY KEY AUTOINCREMENT
                            NOT NULL,
    user_id         INTEGER REFERENCES users (user_id) ON DELETE CASCADE
                            NOT NULL,
    diag_nick       TEXT,
    content         TEXT    NOT NULL,
    role            TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    command         TEXT,
    emotion         TEXT,
    env             TEXT,
    server          TEXT,
    other_info      TEXT,
    bind_to_diag_id INTEGER REFERENCES users_dialogs (diag_id) ON DELETE CASCADE,
    bind_to_nick_id INTEGER REFERENCES users_nicknames (nick_id) ON DELETE SET NULL,
    filter_allowed  INTEGER,
    filter_topics   TEXT
);
""")
    # четвертая с переменными "УМА"
    cursor.execute("""
    CREATE TABLE mind_variables (
    mind_id      INTEGER PRIMARY KEY AUTOINCREMENT
                         UNIQUE
                         NOT NULL,
    mind_name    TEXT,
    mood         REAL    NOT NULL
                         DEFAULT (0.0),
    last_changed TEXT    NOT NULL
);

""")
    db.commit()


class HyperAIDatabase:
    def __init__(self):
        safety_mode = sl.threadsafety  # ДОЛЖНО УКАЗЫВАТЬСЯ ПЕРЕД ИМПОРТОМ!
        # print(f'[DB PRE-INIT] default threadsafety {safety_mode}, attempting to change to 3...')
        # sl.threadsafety = 3
        # https://docs.python.org/3/library/sl.html
        # https://ricardoanderegg.com/posts/python-sqlite-thread-safety/

        if safety_mode == 3:
            self.db_connection = sl.connect('HyperAI_DATABASE.db', check_same_thread=False)
            print("[DB INIT] SUCCESSFULLY CONNECTED TO DATABASE! sl.threadsafety = 3")
        else:
            print(
                f"[DB CANT BE USED!!! RAISING EXCEPTION!!! Почему? Да потому что sl.threadsafety = {str(safety_mode)}")
            raise Exception('**** БАЗА ДАННЫХ В Ж')

        # self.cursor = self.db_connection.cursor()

    def save_db_changes(self) -> bool:
        fail = True
        try_num = 0
        while fail and try_num < 10:
            try_num += 1
            try:
                self.db_connection.commit()
                fail = False
                return True
            except BaseException as err:
                print(f'[DB SAVE ERR] ОШИБКА N={str(try_num)} ПРИ СОХРАНЕНИИ БАЗЫ ДАННЫХ! ', err, )
                print('ТЕКСТ СОХРАНЕНИЯ ОШИБКИ', traceback.format_exc())
                print("\n=== КОНЕЦ ОШИБКИ ====")
                time.sleep(0.1)
        return False

    def exec(self, cursor, sql):
        cursor.execute(sql)
        self.save_db_changes()
        # cursor.close()

    def get_cursor_result(self, cursor: sl.Cursor) -> any:
        row = cursor.fetchone()
        if row is not None:
            if len(row) > 0:
                result = row[0]
                if result is not None:
                    return result
        return None

    def get_cursor_results(self, cursor: sl.Cursor) -> any:
        rows = cursor.fetchall()
        if rows is not None:
            if len(rows) > 0:
                return rows
        return None

    def get_user_id(self, nick: str, cursor=None) -> int:
        ##SELECT ID FROM table_name WHERE City LIKE String
        if cursor is None:
            cursor = self.db_connection.cursor()
        sql = f"""SELECT user_id FROM users_nicknames WHERE nick = ?"""

        cursor.execute(sql, (nick,))
        return self.get_cursor_result(cursor)

    def get_db_field(self, field_id: int, field_name: str, table_name: str = "users", id_name: str = "user_id",
                     many: bool = False, cursor: sl.Cursor = None):
        if cursor is None:
            cursor = self.db_connection.cursor()
        result = None
        try:
            cursor.execute(f"SELECT {field_name} FROM {table_name} WHERE {id_name} = ?", (field_id,))
            if many:
                result = self.get_cursor_results(cursor)
                if result is not None and len(result) <= 0:
                    result = None
            else:
                result = self.get_cursor_result(cursor)
        except BaseException as err:
            print("[DB GETTER FIELD ERR] наверное такого поля нету, ерр:", err)

        return result

    def set_db_field(self, field_id: int, field_name: str, field_new_value: any, table_name: str = "users",
                     id_name: str = "user_id", cursor: sl.Cursor = None) -> bool:
        if cursor is None:
            cursor = self.db_connection.cursor()
        success = False
        try:
            cursor.execute(f"UPDATE {table_name} SET {field_name} = ? WHERE {id_name} = ?",
                           (field_new_value, field_id,))
            if cursor.rowcount >= 1:
                if self.save_db_changes():
                    success = True
            # cursor.execute(f"SELECT {field_name} FROM {table_name} WHERE {id_name} = ?", (field_id,))
        except BaseException as err:
            print("[DB SETTER FIELD ERR] наверное такого поля нету, ерр:", err)
        return success



    def set_mood(self, new_mood: float, mind_id: int = 1) -> bool:
        cursor = self.db_connection.cursor()
        if self.set_db_field(field_id=int(mind_id), field_name="mood", field_new_value=new_mood,
                             table_name="mind_variables", id_name="mind_id", cursor=cursor):
            return self.set_db_field(field_id=int(mind_id), field_name="last_changed",
                                     field_new_value=eztime(), table_name="mind_variables", id_name="mind_id",
                                     cursor=cursor)
        return False

    def get_mood(self, mind_id: int = 1) -> float:
        return self.get_db_field(field_id=int(mind_id), field_name="mood",table_name="mind_variables",id_name="mind_id")

    def get_user_rank(self, user_id: int):
        return self.get_db_field(field_id=int(user_id), field_name="rank")

    def get_user_last_interact_time(self, user_id: int, last_answered=False):
        if last_answered:
            return self.get_db_field(field_id=int(user_id), field_name="last_answered")
        else:
            return self.get_db_field(field_id=int(user_id), field_name="last_interact")

    def get_user_last_question_date(self, user_id: int):
        return self.get_db_field(field_id=int(user_id), field_name="last_question_date")

    def set_user_last_question(self, user_id: int, question: str) -> bool:
        cursor = self.db_connection.cursor()
        return self.set_db_field(field_id=int(user_id), field_name="last_question_date",
                                 field_new_value=eztime(), cursor=cursor) and self.set_db_field(field_id=int(user_id),
                                                                             field_name="last_question",
                                                                             field_new_value=question, cursor=cursor)

    def set_user_last_interact_time(self, user_id: int, new_value: str, last_answered=False):
        if last_answered:
            return self.set_db_field(field_id=int(user_id), field_name="last_answered", field_new_value=new_value)
        else:
            return self.set_db_field(field_id=int(user_id), field_name="last_interact", field_new_value=new_value)

    def set_user_rank(self, user_id: int, new_rank: float):
        return self.set_db_field(field_id=int(user_id), field_name="rank", field_new_value=clamp(new_rank, 0.0, 10.0))

    def add_to_user_rank(self, user_id: int, amount: float):
        rank = self.get_user_rank(user_id)
        if rank is not None:
            result_rank = clamp((amount + rank), 0.0, 10.0)
            if self.set_user_rank(user_id, new_rank=result_rank):
                return result_rank
            else:
                return -1
        else:
            return -1

    def check_user_id_exists(self, user_id: int, cursor: sl.Cursor = None):
        if cursor is None:
            cursor = self.db_connection.cursor()
        cursor.execute("SELECT COUNT(user_id) FROM users WHERE user_id = ?", (user_id,))
        result = self.get_cursor_result(cursor)
        if result:
            return True
        else:
            return False

    def get_or_create_user(self, data: dict, return_nick_id_too: bool = False) -> Union[int, dict, None]:
        nick = data["user"]
        if nick.strip():
            user_id = self.get_user_id(nick)
            if user_id is None:
                user_id = self.add_new_user(data)
            if not return_nick_id_too:
                return user_id
            else:
                return {"user_id": user_id, "nick_id": self.get_user_nick_id(nick, data)}
        return None

    def get_user_nick_id(self, nick: str, data: dict = None, cursor: sl.Cursor = None) -> int:
        if cursor is None:
            cursor = self.db_connection.cursor()
        if data is not None:
            env = data.get("env", None)
            server = data.get("server", None)
            if server and env:
                cursor.execute(
                    """SELECT nick_id FROM users_nicknames WHERE nick = (?) AND env = (?) AND server = (?)""",
                    (nick, env, server))
            elif env:
                cursor.execute("""SELECT nick_id FROM users_nicknames WHERE nick = (?) AND env = (?)""",
                               (nick, env))

        else:
            cursor.execute("""SELECT nick_id FROM users_nicknames WHERE nick = (?)""", (nick,))
        return self.get_cursor_result(cursor)

    def set_analyze_for_nick(self, nick_id: int, nick_analyze:str) -> bool:
        return self.set_db_field(field_id=int(nick_id),field_name="nick_analyze",field_new_value=nick_analyze,table_name="users_nicknames",id_name="nick_id")
    def get_nick_analyze(self, nick_id: int, return_bool: bool):
        if return_bool:
            return True if self.get_db_field(field_id=int(nick_id),field_name="nick_analyze",table_name="users_nicknames",id_name="nick_id") else False
        else:
            return self.get_db_field(field_id=int(nick_id), field_name="nick_analyze", table_name="users_nicknames", id_name="nick_id")

    def add_nick_to_user(self, user_id: int, data: dict) -> int:
        nick = data["user"]
        env = data["env"]
        server = data.get("server", None)
        date = eztime()
        cursor = self.db_connection.cursor()
        sql = f"""
INSERT OR IGNORE INTO users_nicknames (user_id,nick,env,server,other_info,added_date)
VALUES (?,?,?,?,?,?)
RETURNING nick_id
"""
        cursor.execute(sql, (user_id, nick, env, server, None, date,))

        inserted_nick_id = self.get_cursor_result(cursor)
        if inserted_nick_id is not None:
            self.save_db_changes()
            print('[DB ADD NEW NICK TO USER ID(' + str(user_id) + ') INSERTED ' + nick + '. Nick in table ID = (' + str(
                inserted_nick_id) + ')')
        return inserted_nick_id

    def add_new_user(self, data: dict = None) -> Union[int, None]:
        if data == None:
            print("[DB EXCEPTION] **** НАДО ПЕРЕДАТЬ ДАННЫЕ В БАЗУ а DATA пуст!")
            return None
        cursor = self.db_connection.cursor()
        rank = random.choice([3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0])

        date = eztime()
        name = data["user"]
        data["env"] = data.get("env", "minecraft")
        sql = f"""
INSERT INTO users (user_id,name,rank,firstreg,last_interact,diags_count)
VALUES (?,?,?,?,?,?)
RETURNING user_id
f"""
        cursor.execute(sql, (None, name, rank, date, date, 0,))
        inserted_id = self.get_cursor_result(cursor)
        if inserted_id is not None:
            self.save_db_changes()
            print('[DB ADD NEW USER] INSERTED ' + name + ' AT ID ' + str(inserted_id) + ' DEBUG TYPE ' + str(
                type(inserted_id)))
            self.add_nick_to_user(user_id=inserted_id, data=data)  # +ник к юзеру

            if data.get("env", "") == "discord":  # +дискорд ид к юзеру
                discord_user_id = data.get("discord_id", None)
                if discord_user_id:
                    discord_data = dict(data)
                    discord_data["env"] = "discord_id"
                    discord_data["user"] = discord_user_id
                    self.add_nick_to_user(user_id=inserted_id, data=discord_data)
                    print(f'[DB ADD DISCORD NICK] Добавлен для ника {str(name)} discord_id в дискорде')
            elif data.get("env", "") == "youtube":  # +ютуб ид к юзеру
                youtube_user_id = data.get("youtube_user_channel_id", None)
                if youtube_user_id:
                    youtube_data = dict(data)
                    youtube_data["env"] = "youtube_user_channel_id"
                    youtube_data["user"] = youtube_user_id
                    self.add_nick_to_user(user_id=inserted_id, data=youtube_data)
                    print('[DB ADD YT] Добавлен ID в YT')
            elif data.get("env", "") == "trovo":  # +trovo ид к юзеру
                trovo_user_id = data.get("trovo_user_channel_id", None)
                if trovo_user_id:
                    trovo_data = dict(data)
                    trovo_data["env"] = "trovo_user_channel_id"
                    trovo_data["user"] = str(trovo_user_id)
                    self.add_nick_to_user(user_id=inserted_id, data=trovo_data)
                    print('[DB ADD TROVO] Добавлен ID TROVO')

            return inserted_id
        return None

    def add_diags(self, user_id: int, diag_to_add: list[dict], data: dict = None) -> list[int]:
        # log = {"user":user, "role": role, "content": content, "date":eztime(), "emotion":emotion, "command":command}
        cursor = self.db_connection.cursor()
        diag_ids = []
        date = eztime()
        if data is None:
            env = None
            server = None
            other_info = None
            bind_to_nick_id = None
            diag_nick = None
        else:
            def get_other_info_from_env(d):
                result = ""
                sentence_type = d.get("sentence_type", "")
                if sentence_type != "":
                    result += "sentence_type=" + sentence_type

                return result if result else None

            diag_nick = data.get("user", None)
            env = data.get("env", None)
            server = data.get("server", None)

            other_info = get_other_info_from_env(data)  # TODO
            bind_to_nick_id = None  # data.get("user_id", None)
        if bind_to_nick_id == None and diag_nick:
            if diag_nick.strip() != "":
                bind_to_nick_id = self.get_user_nick_id(nick=diag_nick, data=data, cursor=cursor)
        last_user_diag_id = None
        last_interact_time = None  # get last interact
        for i, record in enumerate(diag_to_add):

            filter_allowed = record.get("filter_allowed", None)
            if filter_allowed is not None:
                filter_allowed = int(filter_allowed)
            filter_topics = record.get("filter_topics", None)
            if not filter_topics:
                filter_topics = None

            diag_nick = record.get("user", diag_nick)
            role = record["role"]
            bind_to_diag_id = None
            date_rec = record.get("date", None)
            if role == "assistant" and last_user_diag_id is not None:
                if date_rec is not None:
                    last_interact_time = date_rec
                bind_to_diag_id = last_user_diag_id
            if date_rec is None:
                date_rec = date
            sql = f"""
INSERT INTO users_dialogs (user_id,diag_nick,content,role,date,command,emotion,env,server,other_info,bind_to_diag_id,bind_to_nick_id,filter_allowed,filter_topics)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
RETURNING diag_id
f"""
            cursor.execute(sql, (user_id, diag_nick, record["content"], role, date_rec,
                                 # (user_id,diag_nick,content,          role,  date,
                                 record.get("command", None), record.get("emotion", None),
                                 # command,                   emotion,
                                 env, server, other_info, bind_to_diag_id, bind_to_nick_id,
                                 filter_allowed, filter_topics,
                                 ))
            # env,server,other_info,bind_to_diag_id,bind_to_nick_id)
            inserted_diag_id = self.get_cursor_result(cursor)
            if inserted_diag_id is not None:
                diag_ids.append(inserted_diag_id)
                if role == "user":
                    last_user_diag_id = inserted_diag_id

                # INCREMENT DIAGS COUNT VALUE
                if bind_to_diag_id is not None:
                    cursor.execute('UPDATE users SET diags_count = diags_count + 1 WHERE user_id = ?', (user_id,))
        if len(diag_ids) > 0:
            if last_interact_time:
                cursor.execute('UPDATE users SET last_answered = ? WHERE user_id = ?', (last_interact_time, user_id,))
            if self.save_db_changes():
                return diag_ids
        return []

    def get_user_diags(self, user_id: Union[int, None], count: int = 1):
        # SELECT * FROM l LIMIT 100
        cursor = self.db_connection.cursor()
        user_id_compar = ""
        if user_id is not None:
            user_id_compar = f"user_id = {str(user_id)} AND "
        sql = f"""
SELECT * FROM (
SELECT diag_id,diag_nick,content,role,date,command,emotion,env FROM users_dialogs WHERE diag_id IN
    (SELECT bind_to_diag_id FROM users_dialogs WHERE {user_id_compar}bind_to_diag_id IS NOT NULL AND role = 'assistant')
UNION
    SELECT diag_id,diag_nick,content,role,date,command,emotion,env FROM users_dialogs WHERE {user_id_compar}bind_to_diag_id IS NOT NULL AND role = 'assistant'
ORDER BY diag_id DESC
LIMIT {str(count * 2)}
)
ORDER BY diag_id ASC
"""
        # cursor.execute(sql,(user_id,user_id,count*2,))
        cursor.execute(sql)
        result_diags = self.get_cursor_results(cursor)
        diags = []
        if result_diags is not None:
            for d in result_diags:
                # 0 diag_id,1 diag_nick,2 content,3 role,4 date,5 command,6 emotion,7 env
                # {"user":d[1],"role":d[3],"content":d[2],"date":d[4],"command":d[5],"emotion":d[6],"env":d[7]}
                diag_dict = {}
                field_ids = {"user": 1, "role": 3, "content": 2, "date": 4, "command": 5, "emotion": 6, "env": 7}
                for field, value in field_ids.items():
                    if d[value]:
                        diag_dict[field] = d[value]
                diags.append(diag_dict)
        return diags

    def get_user_diag_count(self, user_id: int, real=True, cursor: sl.Cursor = None):
        if cursor is None:
            cursor = self.db_connection.cursor()
        if real:
            if self.check_user_id_exists(user_id):
                cursor.execute("SELECT COUNT(diag_id) FROM users_dialogs WHERE user_id = ? AND role = 'assistant'",
                               (user_id,))
            else:
                return None
        else:
            cursor.execute("SELECT diags_count FROM users WHERE user_id = ?", (user_id,))
        return self.get_cursor_result(cursor)

    def get_last_any_diag_time(self, cursor: sl.Cursor = None):
        if cursor is None:
            cursor = self.db_connection.cursor()
        cursor.execute("""SELECT * FROM (
SELECT date,diag_id FROM users_dialogs WHERE bind_to_diag_id IS NOT NULL AND role = 'assistant'
ORDER BY diag_id DESC
LIMIT 1
)""")
        return self.get_cursor_result(cursor)

    def get_relevant_diag(self, user_id: Union[int, None] = None, count: int = 1, exact_user_timeout: float = 160.0,
                          any_user_timeout: float = 250.0) -> list[dict]:
        now = datetime.now()
        if user_id is not None:
            last_answered = self.get_db_field(field_id=user_id, field_name="last_answered")
            if last_answered:
                try:
                    if (now - tm(last_answered)).total_seconds() > exact_user_timeout:
                        user_id = None
                except BaseException as err:
                    print(f'[DB Get Time EXACT relevant user_id={str(user_id)} diag ERR] err =', err)
                    user_id = None
            else:
                user_id = None
        diags = []
        if user_id is None:
            last_answered = self.get_last_any_diag_time()
            if last_answered:
                try:
                    if (now - tm(last_answered)).total_seconds() <= any_user_timeout:
                        print('[DB get ANY RELEVANT DIAG] time succed, searching for ANY diag..')
                        diags = self.get_user_diags(user_id=None, count=count)
                except BaseException as err:
                    print('[DB Get Time ANY relevant diag ERR] err =', err)
        else:
            print(f'[DB get EXACT user_id={str(user_id)} RELEVANT] time succed, searching for EXACT diag..')
            diags = self.get_user_diags(user_id=user_id, count=count)
        return diags

    def connection_close(self):
        self.db_connection.close()


if __name__ == "__main__":
    db = HyperAIDatabase()
    # db.add_nick_to_user(4, {"user": "LexaLepexa", "env": "youtube"})
    # debugChatEntry = {"user": "LexaLepaxa2324", "env": "minecraft", "server": "mc.musteryworld.me", "date": eztime()}
    # id = db.get_or_create_user(debugChatEntry)
    ## id = db.get_user_id("NetTyan")
    # print("id = " + str(id), type(id))
    # curs = db.db_connection.cursor()
    # curs.execute("SELECT * FROM users_nicknames WHERE nick LIKE ?", ("LexaLepexa",))

    # print('lol', db.get_cursor_results(curs), 'lol')
    debug_dialog = [
        {"user": "LexaLepexa", "role": "user", "content": "2Привет! Как дела?", "date": "2023-06-30 20:22:59",
         "emotion": "", "command": ""},
        {"user": "default", "role": "assistant", "content": "3Пока! Ты скучный!", "date": "2023-06-30 20:23:00",
         "emotion": "агрессия", "command": "бан"},
        {"user": "LexaLepexa", "role": "user", "content": "4ЭЭэээ", "date": "2023-06-30 20:24:59",
         "emotion": "", "command": ""},
        {"user": "default", "role": "assistant", "content": "5Не эээкай нищщщ", "date": "2023-06-30 20:25:00",
         "emotion": "агрессия", "command": "бан"},
    ]
    # db.add_diags(user_id=db.get_user_id("LexaLepexa"),diag_to_add=debug_dialog)
    # print(db.get_user_nick_id("LexaLepexa",{"env":"youtube"}))
    # print(db.get_user_diag_count(user_id=db.get_user_id("Net Tyan"),real=True))#db.get_user_diags(user_id=None, count=1))
    # print(db.set_db_field(field_id=4,field_new_value=None,field_name="other_info", table_name="users_nicknames", id_name="user_id"))
    # print(db.get_last_any_diag_time())
    # print(db.get_relevant_diag(user_id=db.get_user_id("LexaLepexa"),count=1,exact_user_timeout=1000000.0,any_user_timeout=10000000.0))
    #print(db.add_to_user_rank(user_id=7, amount=-3.2))
    print(db.set_mood(-5))
    print(db.get_mood())

    db.connection_close()
    # print('Debug mode actived')
    # db = sl.connect('HyperAI_DATABASE.db')
    # cursor = db.cursor()
    # db_add_new_user()
    ##db_init(db, cursor)
    # db.close()
