# coding: utf-8
import torch
from animal import transform, Net # animal.py から
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64

#  推論
def predict(img):
    net = Net().cpu().eval()
    net.load_state_dict(torch.load('./dog_cat_data.pt', map_location=torch.device('cpu')))
    img = transform(img)
    img = img.unsqueeze(0)
    #　推論
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y


#　犬か猫か返す関数
def getName(label):
    if label==0:
        return '猫'
    elif label==1:
        return '犬'


app = Flask(__name__)

# アップロード 拡張子制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

#　拡張子チェック
def allwed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'filename' not in request.files:
            return redirect(request.url)
        # データの取り出し
        file = request.files['filename']
        # ファイルのチェック
        if file and allwed_file(file.filename):

            #　画像ファイルに対する処理
            buf = io.BytesIO()
            image = Image.open(file).convert('RGB')
            image.save(buf, 'png')
            #　バイナリデータbase64エンコード utf-8デコード
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            #　HTML側src付帯情報
            base64_data = 'data:image/png;base64,{}'.format(base64_str)

            # 入力された画像推論
            pred = predict(image)
            animalName_ = getName(pred)
            return render_template('result.html', animalName=animalName_, image=base64_data)
        return redirect(request.url)

    # GET メソッド
    elif request.method == 'GET':
        return render_template('index.html')

    
# アプリケーション実行
if __name__ == '__main__':
    app.run(debug=True)
