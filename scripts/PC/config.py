# Confurações da câmera
camera_id = 0         # Defina o id da câmera (Padrão: 0)
camera_exposure = -8  # Defina exposição da câmera (Padrão: -8)

# Multiplicadores de deslocamento
deltax_multiplier = 1 #Defina o multiplicador de deslocamento X
deltay_multiplier = 1 #Defina o multiplicador de deslocamento Y

# Configurações de estimativa
border_black_windowing = True  # Aplica o escurecimento nas bordas das imagens (Padrão: True)
phase_windowing = None  # Aplica o janelamento no sinal final da fase (None, "initial", "central") (Padrão: None)

# Configure uma porta COM, deixe None para tentar encontrar automaticamente
usb_com_port = None  # Deve ser None ou uma string, exemplo: "COM4" (veja o arquivo "config_tools/usb_list.py") (Valor padrão: True)

