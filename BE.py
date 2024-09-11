from flask import Flask, render_template, request, Response, jsonify
import subprocess
import os
from flask import Flask, jsonify, send_from_directory, abort

app = Flask(__name__)

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    def generate():
        for line in process.stdout:
            yield f"{line}<br/>\n"
        process.wait()
    return Response(generate(), mimetype='text/html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Gaussian-DP', methods=['GET', 'POST'])
def script1():
    if request.method == 'POST':
        args = {
            '--lr': request.form.get('lr', '0.001'),
            '--epochs': request.form.get('epochs', '100'),
            '--bs': request.form.get('bs', '32'),
            '--eps': request.form.get('eps', '2'),
            '--grad_norm': request.form.get('grad_norm', '0.1')
        }
        command = ['python', '-u', r'C:\Users\18521\Downloads\DO_AN\private_vision\Gaussian-DP.py']
        for arg, value in args.items():
            if value != 'False':
                command.append(arg)
                if value != 'True':
                    command.append(value)
        return run_command(command)
    return render_template('temp.html')

@app.route('/top_n_selction', methods=['GET', 'POST'])
def script2():
    if request.method == 'POST':
        args = {
            '--model': request.form.get('model', 'DP_SGD'),
            '--data_name': request.form.get('data_name', 'celeba'),
            '--top_n': request.form.get('top_n'),
            '--num_classes': request.form.get('num_classes', '1000'),
            '--save_root': request.form.get('save_root', 'reclassified_public_data'),
            '--bs': request.form.get('bs', '32'),
            '--eps': request.form.get('eps', '2')
        }
        command = ['python', '-u', r'C:\Users\18521\Downloads\DO_AN\LPG-MI\top_n_selection.py']
        for arg, value in args.items():
            if value != 'False':
                command.append(arg)
                if value != 'True':
                    command.append(value)
        return run_command(command)
    return render_template('top_n_selction.html')

@app.route('/TrainCgain', methods=['GET', 'POST'])
def script3():
    if request.method == 'POST':
        args = {
            '--data_root': request.form.get('data_root'),
            '--data_name': request.form.get('data_name'),
            '--target_model': request.form.get('target_model'),
            '--private_data_root': request.form.get('private_data_root', './datasets/celeba_private_domain'),
            '--eps': request.form.get('eps', '2'),
            '--batch_size': request.form.get('batch_size', '32'),
            '--eval_batch_size': request.form.get('eval_batch_size'),
            '--gen_num_features': request.form.get('gen_num_features', '64'),
            '--gen_dim_z': request.form.get('gen_dim_z', '128'),
            '--gen_bottom_width': request.form.get('gen_bottom_width', '4'),
            '--gen_distribution': request.form.get('gen_distribution', 'normal'),
            '--dis_num_features': request.form.get('dis_num_features', '64'),
            '--lr': request.form.get('lr', '0.0002'),
            '--beta1': request.form.get('beta1', '0.0'),
            '--beta2': request.form.get('beta2', '0.9'),
            '--seed': request.form.get('seed', '46'),
            '--max_iteration': request.form.get('max_iteration', '30000'),
            '--n_dis': request.form.get('n_dis', '5'),
            '--num_classes': request.form.get('num_classes', '1000'),
            '--loss_type': request.form.get('loss_type', 'hinge'),
            '--relativistic_loss': request.form.get('relativistic_loss'),
            '--calc_FID': request.form.get('calc_FID'),
            '--results_root': request.form.get('results_root', 'results'),
            '--no_tensorboard': request.form.get('no_tensorboard'),
            '--no_image': request.form.get('no_image'),
            '--checkpoint_interval': request.form.get('checkpoint_interval', '200'),
            '--log_interval': request.form.get('log_interval', '100'),
            '--eval_interval': request.form.get('eval_interval', '1000'),
            '--n_eval_batches': request.form.get('n_eval_batches', '100'),
            '--n_fid_images': request.form.get('n_fid_images', '3000'),
            '--args_path': request.form.get('args_path'),
            '--gen_ckpt_path': request.form.get('gen_ckpt_path'),
            '--dis_ckpt_path': request.form.get('dis_ckpt_path'),
            '--alpha': request.form.get('alpha', '0.2'),
            '--inv_loss_type': request.form.get('inv_loss_type', 'margin'),
            '--cGAN': request.form.get('cGAN')
        }

        command = ['python', '-u', r'C:\Users\18521\Downloads\DO_AN\LPG-MI\train_cgan.py']
        for arg, value in args.items():
            if value is not None and value != 'False':
                command.append(arg)
                if value != 'True':
                    command.append(value)

        return run_command(command)

    return render_template('TrainCgain.html')

@app.route('/reconstruct_cpu', methods=['GET', 'POST'])
def run_script4():
    if request.method == 'POST':
        model = request.form.get('model', 'VGG16')
        inv_loss_type = request.form.get('inv_loss_type', 'margin')
        lr = request.form.get('lr', 0.1)
        iter_times = request.form.get('iter_times', 600)
        gen_num_features = request.form.get('gen_num_features', 64)
        gen_dim_z = request.form.get('gen_dim_z', 128)
        gen_bottom_width = request.form.get('gen_bottom_width', 4)
        gen_distribution = request.form.get('gen_distribution', 'normal')
        save_dir = request.form.get('save_dir', 'PLG_MI_Inversion')
        path_G = request.form.get('path_G', '')

        command = [
            'python', '-u', r'C:\Users\18521\Downloads\DO_AN\LPG-MI\reconstruct_cpu.py',
            '--model', model,
            '--inv_loss_type', inv_loss_type,
            '--lr', str(lr),
            '--iter_times', str(iter_times),
            '--gen_num_features', str(gen_num_features),
            '--gen_dim_z', str(gen_dim_z),
            '--gen_bottom_width', str(gen_bottom_width),
            '--gen_distribution', gen_distribution,
            '--save_dir', save_dir,
            '--path_G', path_G
        ]
        return run_command(command)

    return render_template('reconstruct_cpu.html')

@app.route('/files/folder1')
def get_folder1_files():
    folder_path = r'C:\Users\18521\Downloads\DO_AN\files\folder1'
    files = os.listdir(folder_path)
    return jsonify(files)

@app.route('/files/folder2')
def get_folder2_files():
    folder_path = r'C:\Users\18521\Downloads\DO_AN\files\folder2'
    files = os.listdir(folder_path)
    return jsonify(files)
@app.route('/summary')
def ty():
    return render_template('summary.html')
@app.route('/files/<folder>/<filename>')
def serve_file(folder, filename):
    base_path = r'C:\Users\18521\Downloads\DO_AN\files'
    
    # Ensure the folder is either 'folder1' or 'folder2'
    if folder not in ['folder1', 'folder2']:
        print(f"Invalid folder: {folder}")
        abort(404)
    
    # Construct the full file path
    folder_path = os.path.join(base_path, folder)
    file_path = os.path.join(folder_path, filename)

    print(f"Looking for file at: {file_path}")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        abort(404)

    try:
        # Serve the file if it exists
        return send_from_directory(folder_path, filename)
    except Exception as e:
        print(f"Error serving file: {e}")
        abort(500)
@app.route('/files/<folder>')
def get_files(folder):
    folder_paths = {
        'folder1': r'C:\Users\18521\Downloads\DO_AN\files\folder1',
        'folder2': r'C:\Users\18521\Downloads\DO_AN\files\folder2'
    }
    folder_path = folder_paths.get(folder)
    if not folder_path:
        return jsonify([])

    try:
        files = os.listdir(folder_path)
        return jsonify(files)
    except FileNotFoundError:
        return jsonify([])

@app.route('/files/<folder>/<filename>')
def get_file(folder, filename):
    folder_paths = {
        'folder1': r'C:\Users\18521\Downloads\DO_AN\files\folder1',
        'folder2': r'C:\Users\18521\Downloads\DO_AN\files\folder2'
    }
    folder_path = folder_paths.get(folder)
    if not folder_path:
        return "Folder not found", 404

    try:
        return send_from_directory(folder_path, filename)
    except FileNotFoundError:
        return "File not found", 404


@app.route('/delete/<folder>/<filename>', methods=['DELETE'])
def delete_file(folder, filename):
    base_dir = os.path.join('files', folder)
    extensions = ['.txt', '.png', '.tar']  # List all extensions to delete
    success = True

    for ext in extensions:
        file_path = os.path.join(base_dir, filename + ext)  # Correctly append extension once
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        else:
            success = False
            print(f"File not found: {file_path}")
    print(f"Looking for file at: {file_path}")

    if success:
        return jsonify({'message': 'Files deleted successfully'}), 200
    else:
        return jsonify({'message': 'Some files could not be deleted'}), 404

if __name__ == '__main__':
    app.run(debug=True)



