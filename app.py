import io
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from models import Generator
import numpy as np
import cv2
import pandas as pd

# Streamlitアプリケーションの定義
st.title('パラドックスポケモン生成AI')
st.caption("画像のアップロードかポケモン名を入力してください")
st.caption("リージョンフォルムの場合などはポケモン名の後に(地方名のすがた)と入力してください。")

#ポケモン名の入力
poke_name=st.text_input("ポケモン名を入力してください")

#画像のアップロード
uploaded_file = st.file_uploader('変換する画像をアップロードしてください', type=['png',"jpg","jpeg"])

#deviceの定義
device = torch.device('cpu')

#使用モデルの選択
version=st.selectbox("どっちの姿?",["古代の姿","未来の姿"])

if version=="古代の姿":
# 学習したCycleGANのモデルのパス
    G_AB_path = './model/netG_N2P.pth'
else:
    G_AB_path = './model/netG_N2F.pth'

# 保存したモデルを呼び出し、自前の画像を変換
G_AB = Generator(input_nc=3, output_nc=3)
G_AB.load_state_dict(torch.load(G_AB_path,map_location=device))

# 画像変換の関数の定義
def transform_image(image):
    transform = transforms.Compose([
    transforms.Resize(int(256),Image.BICUBIC),
    transforms.CenterCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

    # 透過部分を白塗りにするための処理
    np_image = np.array(image)
    alpha_channel = np_image[:, :, 3]
    np_image[alpha_channel == 0] = [255, 255, 255, 255]
    image = Image.fromarray(np_image)
    image = image.convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 変換器をモデルに渡す
    with torch.no_grad():
        transformed_image = G_AB(image_tensor)
    
    # Tensorを画像に変換
    transformed_image = transformed_image.squeeze(0).detach().cpu()
    transformed_image = (transformed_image + 1.0) / 2.0
    transformed_image = transforms.ToPILImage()(transformed_image)

    return transformed_image

with st.form(key='profile form'):
    #ボタン
    submit_btn=st.form_submit_button("生成")
    cancel_btn=st.form_submit_button("リセット")
    if uploaded_file is not None  and submit_btn:
        # アップロードされた画像を読み込む
        image = Image.open(io.BytesIO(uploaded_file.read()))
        image=image.convert("RGBA")

        # 画像を変換する
        transformed_image = transform_image(image)

        # 変換後の画像を表示する
        st.image(transformed_image, caption='変換後の画像', use_column_width=True)

    elif uploaded_file is  None and poke_name is not None and submit_btn:
        # ポケモン名の画像を読み込む
        df=pd.read_csv("./image/all_data.csv",index_col=0)
        #画像の番号の取得
        number=df[df["ポケモン名"]==poke_name].index
        try:
            number=number[0]
            image = Image.open("./image/"+str(number)+".jpg")
            image=image.convert("RGBA")
            # 画像を変換する
            transformed_image = transform_image(image)

            # 変換後の画像を表示する
            st.image(transformed_image, caption='変換後の画像', use_column_width=True)
        except IndexError:
            st.write(f"入力したポケモンは存在しません!")
