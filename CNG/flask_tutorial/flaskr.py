from sqlite3 import dbapi2 as sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash, _app_ctx_stack

# configuration
DATABASE = 'flaskr.db'
DEBUG = True
SECRET_KEY = 'development key'
USERNAME = 'admin'
PASSWORD = 'default'

# create our little application :)
app = Flask(__name__) # Flask 객체 생성
app.config.from_object(__name__) # 위의 환경변수들을 읽어 Flask의 환경 변수에 적용
# app.config.from_envvar('FLASKR_SETTINGS', silent=True)

# DB초기화 함수 
def init_db():
    # DB 테이블 생성
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f: # app객체의 함수 : 리소스경로의 파일 열고 값 읽기
            db.cursor().executescript(f.read())
        db.commit()

def get_db():
    # 연결이 없는 경우 새 DB연결
    top = _app_ctx_stack.top
    if not hasattr(top, 'sqlite_db'):
        sqlite_db = sqlite3.connect(app.config["DATABASE"]) # DB연결
        sqlite_db.row_factory = sqlite3.Row
        top.sqlite_db = sqlite_db

    return top.sqlite_db


@app.teardown_appcontext
def close_db_connection(exception):
    # 요청이 끝나면 DB 다시 닫기
    top = _app_ctx_stack.top
    if hasattr(top, 'sqlite_db'):
        top.sqlite_db.close()

# @app.route('/') url 홈에 호출시 아래 함수 실행
@app.route('/')
def show_entries(): # 작성된 글 보여주기 
    db = get_db()
    cur = db.execute('select title, text from entries order by id desc')
    entries = cur.fetchall()
    return render_template('show_entries.html', entries=entries)


@app.route('/add', methods=['POST'])
def add_entry(): # 새로운 글 추가 
    if not session.get('logged_in'):
        abort(401)
    db = get_db()
    db.execute('insert into entries (title, text) values (?, ?)',
                 [request.form['title'], request.form['text']]) # title, text db서버에 보내기
    db.commit() 
    flash('New entry was successfully posted')
    return redirect(url_for('show_entries'))


@app.route('/login', methods=['GET', 'POST'])
def login(): # 로그인
    error = None
    if request.method == 'POST': # POST 방식
        if request.form['username'] != app.config['USERNAME']:
            error = 'Invalid username'
        elif request.form['password'] != app.config['PASSWORD']:
            error = 'Invalid password'
        else: # 로그인 성공시 
            session['logged_in'] = True
            flash('You were logged in')
            return redirect(url_for('show_entries')) # entries를 보여줌
    return render_template('login.html', error=error)


@app.route('/logout')
def logout(): # 로그아웃 
    session.pop('logged_in', None)
    flash('You were logged out')
    return redirect(url_for('show_entries'))


if __name__ == '__main__':
    init_db()
    app.run() # 서버 가동