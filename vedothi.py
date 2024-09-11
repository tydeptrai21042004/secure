import matplotlib.pyplot as plt
import sys

def plot_metrics(file_path, output_path):
    # Đọc file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Khởi tạo các danh sách để lưu trữ dữ liệu
    epochs = []
    train_losses = []
    train_accs = []
    test_accs = []

    # Trích xuất dữ liệu từ các dòng liên quan
    for line in lines:
        if line.startswith('Epoch'):
            parts = line.split('\t')
            epoch = int(parts[0].split(':')[1])
            train_loss = float(parts[2].split(':')[1])
            train_acc = float(parts[3].split(':')[1])
            test_acc = float(parts[4].split(':')[1])

            epochs.append(epoch)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

    # Thiết lập kích thước của toàn bộ figure và cỡ chữ
    plt.figure(figsize=(5, 5))  # Giảm kích thước figure xuống 6x6
    plt.rcParams.update({'font.size': 6})  # Giảm cỡ chữ xuống

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy', color='green')
    plt.plot(epochs, test_accs, label='Test Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # Lưu plot dưới dạng file PNG
    plt.savefig(output_path, format='png')
    print(f"Plot saved as {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_metrics.py <file_path> <output_path>")
    else:
        plot_metrics(sys.argv[1], sys.argv[2])
