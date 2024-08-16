import subprocess
import os

def find_usb_drive():
    try:
        # Tenta desmontar o pendrive se estiver montado
        try:        
            command = ["sudo", "umount", "/media/my32gb"]
            subprocess.run(command, check=True)
            print("Pendrive desmontado com sucesso.")
        except subprocess.CalledProcessError:
            print("Pendrive não estava montado ou erro ao desmontar.")

        # Executa o comando lsblk para listar os dispositivos de bloco
        output = subprocess.check_output(['lsblk', '-o', 'NAME,TYPE'], text=True)
        lines = output.strip().split('\n')

        # Identifica o dispositivo USB
        for line in lines:
            print(line)
            if 'disk' in line:
                name = line.split()[0]
                print(name)
                device = "/dev/" + name
                print(f"Dispositivo USB encontrado: {device}")

                # Monta o dispositivo
                command = ["sudo", "mount", "-t", "vfat", "-o", "rw", device, "/media/my32gb"]
                try:
                    subprocess.run(command, check=True)
                    print("Pendrive montado com sucesso.")
                    
                    # Ajusta as permissões do diretório de montagem
                    subprocess.run(["sudo", "chmod", "777", "/media/my32gb"], check=True)
                    
                    return "/media/my32gb"
                except subprocess.CalledProcessError as e:
                    print(f"Erro ao montar o pendrive: {e}")
                    return None

        print("Nenhum dispositivo USB encontrado.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar lsblk: {e}")
        return None

def copy_file_to_usb(src_file, dest_dir):
    if not os.path.exists(src_file):
        print(f"Arquivo {src_file} não encontrado.")
        return

    if not os.path.isdir(dest_dir):
        print(f"Diretório de destino {dest_dir} não encontrado.")
        return

    dest_file = os.path.join(dest_dir, os.path.basename(src_file))

    try:
        # Copia o arquivo para o pendrive usando sudo
        command = ["sudo", "cp", src_file, dest_file]
        subprocess.run(command, check=True)
        print(f"Arquivo {src_file} copiado para {dest_file} com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"Erro ao copiar o arquivo: {e}")

def main():
    # Nome do arquivo a ser copiado
    src_file = "test.txt"

    # Encontra e monta o pendrive
    usb_path = find_usb_drive()
    if usb_path:
        # Copia o arquivo para o pendrive
        copy_file_to_usb(src_file, usb_path)
        
        # Desmonta o pendrive após a cópia
        try:
            command = ["sudo", "umount", usb_path]
            subprocess.run(command, check=True)
            print("Pendrive desmontado com sucesso.")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao desmontar o pendrive: {e}")

if __name__ == "__main__":
    main()
