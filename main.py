from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim

# 変換
data_tens = transforms.ToTensor()
batch_size = 512 # 学習時に、一度に処理するデータの数

# データセットの準備
train_data = torchvision.datasets.FashionMNIST(
    './datasets', train=True, download=True, transform=data_tens
)
test_data = torchvision.datasets.FashionMNIST(
    './datasets', train=False, download=True, transform=data_tens
)

train_dataloader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True
)

classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandals', 'Shirt', 'Sneaker', 'Bag', 'Ankle boots')

# クラスごとの画像を1枚ずつ表示
def show_examples():
    class_ids, sample_indices = np.unique(train_data.targets, return_index=True)
    fig = plt.figure(figsize=(10, 4))
    fig.suptitle("Examples of every class in the Fashion-MNIST dataset", fontsize="x-large")

    for i in class_ids:
        img = train_data.data[sample_indices[i]]
        class_name = train_data.classes[i]
        ax = fig.add_subplot(2, 5, i + 1)
        ax.set_title(f"{i}: {class_name}")
        ax.set_axis_off()
        ax.imshow(img, cmap="gray")
    plt.show()

# モデル定義
class Net(nn.Module):
    def __init__(self):
        super().__init__() # nn.Moduleの初期化
        # データの用意
        self.features = nn.Sequential( # 順番に処理をするコンテナ
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # (白黒は1/カラーは3, 特徴マップの数, 画像を見る倍率(3*3), 余白(詳細は次行)(, stride(通常は1)おきにフィルタを動かす))
            # 出力サイズ = ((入力サイズ + padding*2 - kernel_size)/stride) + 1、今回はpadding=1とすると入力サイズと出力サイズが等しい
            nn.ReLU(), # 負の値は0にして、正の値のみ通す
            nn.MaxPool2d(kernel_size=2), # 32枚に増えた画像それぞれの特徴的な部分だけを抜き出す、2*2の領域ごとに特徴を抜き出すためサイズはもとの1/2
            # 畳み込み層、作成した32枚の特徴マップから64枚の特徴マップに
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # データの分類
        self.classifier = nn.Sequential(
            nn.Flatten(), # 畳み込みの出力を全結合層に渡せる形に変換
            nn.Dropout(0.5), # 過学習を防ぐためランダムに一部のニューロンを無効化する、デフォルトは50% nn.Dropout(p = 0.5)
            nn.Linear(64 * 7 * 7, 128), # (64枚 * 28/2/2 * 28/2/2, 出力チャンネル数)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10), # FashionMNISTは10クラス分類なので出力を10個に
            nn.LogSoftmax(dim=1), # 10個の出力を、log(確率)に変換する活性化関数
        )

    def forward(self, x):
        x = self.features(x)  # 畳み込み＋プーリングで特徴抽出
        x = self.classifier(x)  # 全結合＋Dropout＋分類
        return x

# 訓練関数
def train(model, device, data_loader, optim):
    model.train()  # モデルを訓練モードに設定（Dropout などが有効になる）

    total_loss = 0        # 全バッチの損失の合計
    total_correct = 0     # 正解数の合計

    # データローダーからバッチ単位でデータを取得
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)  # デバイス（CPU or GPU）に転送

        output = model(data)  # モデルにデータを入力し、出力を得る（順伝播）

        loss = nll_loss(output, target)  # 損失関数（NLLLoss）で出力と正解を比較して誤差を計算
        total_loss += float(loss)        # 損失を合計に加算

        optim.zero_grad()    # 勾配を初期化（前回の計算結果をクリア）
        loss.backward()      # 誤差逆伝播で勾配を計算
        optim.step()         # 重みを更新

        pred_target = output.argmax(dim=1)  # 出力の最大値のインデックス（=予測クラス）を取得
        total_correct += int((pred_target == target).sum())  # 正解数を加算

    # データ全体に対する平均損失と正解率を計算
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = total_correct / len(data_loader.dataset)

    return avg_loss, accuracy  # 平均損失と精度を返す


# テスト関数
def test(model, device, data_loader):
    # モデルを評価モードに設定（DropoutやBatchNormなどを無効にする）
    model.eval()

    # 評価時は勾配を計算しないことで高速化＆メモリ節約
    with torch.no_grad():
        total_loss = 0  # 累積損失
        total_correct = 0  # 正解数の累積

        # テストデータローダーからバッチごとにデータを取得
        for data, target in data_loader:
            # データと正解ラベルを使用デバイス（CPUまたはGPU）に転送
            data, target = data.to(device), target.to(device)

            # モデルにデータを入力して予測結果を得る
            output = model(data)

            # 損失（誤差）を計算（グローバルで定義されたnll_lossを使用）
            loss = nll_loss(output, target)
            total_loss += float(loss)  # 損失を合計に加算

            # 出力結果から最もスコアが高いクラスを予測ラベルとして取得
            pred_target = output.argmax(dim=1)

            # 予測ラベルと正解ラベルが一致した数をカウントして加算
            total_correct += int((pred_target == target).sum())

    # 平均損失をサンプル数で割って計算
    avg_loss = total_loss / len(data_loader.dataset)

    # 正解率（accuracy）をサンプル数で割って計算
    accuracy = total_correct / len(data_loader.dataset)

    # 平均損失と正解率を返す
    return avg_loss, accuracy


# メイン処理
def main():
    # クラスごとの画像を表示する関数（必須ではないので省略可能）
    show_examples()

    # 使用するデバイスを決定：CUDA(GPU)が利用可能ならGPU、なければCPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用するデバイス：", device)

    # モデルを初期化して選択したデバイスに配置
    model = Net().to(device)

    # グローバル変数として損失関数を定義（学習・テスト関数からアクセスするため）
    global nll_loss
    nll_loss = nn.NLLLoss()

    # 最適化アルゴリズムをAdamに設定し、学習率を指定
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 学習設定：最大エポック数、Early Stopping の待機エポック数
    n_epochs = 50
    patience = 10  # 改善が見られない場合に学習を打ち切るまでのエポック数
    best_accuracy = 0  # 最良のテスト精度
    wait = 0  # 精度が更新されなかったエポック数カウント
    history = defaultdict(list)  # 学習履歴（損失・精度）を記録する辞書

    # 学習ループ
    for epoch in range(n_epochs):
        # 1エポック分の学習（train関数）
        train_loss, train_accuracy = train(model, device, train_dataloader, optimizer)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        # テストデータでの性能評価（test関数）
        test_loss, test_accuracy = test(model, device, test_dataloader)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        # 結果表示
        print(
            f"epoch {epoch + 1} "
            f"[train] loss: {train_loss:.6f}, accuracy: {train_accuracy:.0%} "
            f"[test] loss: {test_loss:.6f}, accuracy: {test_accuracy:.0%}"
        )

        # Early Stopping の処理：精度が改善されたかチェック
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            wait = 0  # 待機エポック数をリセット
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break  # 改善が一定回数見られなければ学習打ち切り

    # 学習履歴のグラフ表示（損失と精度）
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 3))

    # 損失の推移グラフ
    ax1.set_title("Loss")
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["test_loss"], label="test")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    # 精度の推移グラフ
    ax2.set_title("Accuracy")
    ax2.plot(epochs, history["train_accuracy"], label="train")
    ax2.plot(epochs, history["test_accuracy"], label="test")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    # グラフを表示
    plt.show()


# Windows環境でマルチプロセッシングが正しく動作するようにする
if __name__ == "__main__":
    main()
