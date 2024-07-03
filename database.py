import sqlite3
from tkinter import messagebox

def create_table():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = "DROP TABLE IF EXISTS login"
    cursor.execute(query)
    conn.commit()

    query = "CREATE TABLE login(username VARCHAR UNIQUE, password VARCHAR, permission VARCHAR)"
    cursor.execute(query)
    conn.commit()


def create_table_2():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = "DROP TABLE IF EXISTS user_details"
    cursor.execute(query)
    conn.commit()

    query = "CREATE TABLE user_details(username VARCHAR, email VARCHAR, mobile_number VARCHAR, number_plate VARCHAR,  college VARCHAR,  department VARCHAR,  class VARCHAR)"
    cursor.execute(query)
    conn.commit()


def create_table_3():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = "DROP TABLE IF EXISTS entries"
    cursor.execute(query)
    conn.commit()

    query = "CREATE TABLE entries(username VARCHAR, number_plate VARCHAR, entry_time INTEGER)"
    cursor.execute(query)
    conn.commit()

# login 
def add_user(username, password, permission):
        
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = "INSERT INTO login (username, password, permission) VALUES (?, ?, ?)"
    try:
        cursor.execute(query, (username, password, permission))
    except:
        reponse = messagebox.showerror("Error", "This username already exists\n Please enter other username")
    conn.commit()
    cursor.close()
    conn.close()

def add_user_details(username, vehicle_num, mobile_num, email, college, dept, cls):
        
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = "INSERT INTO user_details (username, email, mobile_number, number_plate, college, department, class) VALUES (?, ?, ?, ?, ?, ?, ?)"
    try:
        cursor.execute(query, (username, email, mobile_num, vehicle_num, college, dept, cls))
    except:
        reponse = messagebox.showerror("Error", "This username already exists\n Please enter other username")
    conn.commit()
    cursor.close()
    conn.close()


def add_entry(username, vehicle_num_plate, time):
    
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = "INSERT INTO entries (username, number_plate, entry_time) VALUES (?, ?, ?)"
    try:
        cursor.execute(query, (username, vehicle_num_plate, time))
    except:
        reponse = messagebox.showerror("Error", "This username already exists\n Please enter other username")
    conn.commit()
    cursor.close()
    conn.close()



def check_user(username, password, permission):
        
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = 'SELECT * FROM login WHERE username = ? AND password = ? AND permission = ?'
    cursor.execute(query, (username, password, permission))
    result = cursor.fetchone()
    conn.commit()
    cursor.close()
    conn.close()
    return result


def check_if_user_details_exist(username):
        
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = 'SELECT * FROM user_details WHERE username = ?'
    cursor.execute(query, [username])
    result = cursor.fetchone()
    conn.commit()
    cursor.close()
    conn.close()
    return result


def update_password(username, password, permission):
        
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = 'UPDATE login SET password = ? where username = ? and permission = ? '
    cursor.execute(query, (password, username, permission))
    conn.commit()
    cursor.close()
    conn.close()


def get_all_number_plates():
        
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = 'SELECT username, email, mobile_number, number_plate FROM "user_details"; '
    cursor.execute(query, ())
    result = cursor.fetchall()
    conn.commit()
    cursor.close()
    conn.close()
    return result

def get_last_5_records():
        
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    # query = 'SELECT * from entries order by entry_time desc limit 5'
    query = 'SELECT a.*, b.college, b.department, b.class from ( ( select username, number_plate, entry_time from entries order by entry_time desc limit 5 )a join ( select * FROM user_details )b  on a.username = b.username );'
    
    cursor.execute(query, ())
    result = cursor.fetchall()
    conn.commit()
    cursor.close()
    conn.close()
    return result

def get_user_type():
    
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = 'SELECT * from entries order by entry_time desc limit 5'
    cursor.execute(query, ())
    result = cursor.fetchall()
    conn.commit()
    cursor.close()
    conn.close()
    return result


# add_user("a","a","teacher")

# create_table_2()
# add_user_details('assdf','a','a','a','a','a','a')
# print(get_last_5_records())

# create_table_3()