from flask import Flask, render_template
import psycopg2

class MyApp:
    def __init__(self):
        self.app = Flask(__name__) # flask로 app객체 생성
        self.POSTGRES_USER = "genia"
        self.POSTGRES_DATABASE = "genia"
        self.POSTGRES_PASSWORD = "test"
        self.conn = None

        # 데이터베이스 연결
        self.connect_database()

        # 라우트 등록
        self.register_routes()

    def connect_database(self): # DB 연결 
        self.conn = psycopg2.connect(
            host = 'localhost',
            dbname = self.POSTGRES_DATABASE,
            user = self.POSTGRES_USER,
            password = self.POSTGRES_PASSWORD,
            port = 5431
        )

    def register_routes(self):
        # Flask 애플리케이션에 루트 경로(/)를 등록
        self.app.route('/')(self.index)

    def index(self):
        # 데이터베이스에서 데이터 가져오기
        sql = "SELECT * FROM video_meta"
        cur = self.conn.cursor()
        cur.execute(sql)
        data = cur.fetchall()
        cur.close()

        # 템플릿에 데이터 전달하고 렌더링
        return render_template('home.html', data=data)

    def run(self):
        # Flask app을 실행하는 메서드 
        self.app.run(debug=True)

if __name__ == '__main__':
    my_app = MyApp() # MyApp 인스턴스 생성
    my_app.run() # 실행