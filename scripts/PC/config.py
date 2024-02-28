# Configurações de câmera
camera_id = 1          # Defina o id da câmera (Padrão: 0, Valores Possíveis: Inteiros)
camera_exposure = -8   # Defina exposição da câmera (Padrão: -8, Valores Possíveis: Inteiros)

# Multiplicadores de deslocamento
deltax_multiplier = 1 # Defina o multiplicador de deslocamento X (Padrão: 1, Valores Possíveis: Inteiros)
deltay_multiplier = 1 # Defina o multiplicador de deslocamento Y (Padrão: 1, Valores Possíveis: Inteiros)

# Configuração de comunicação Serial
usb_com_port = None  # Configure a porta de comunicação serial (Padrão: None, Valores Possíveis: String, Exemplo: "COM4")

# Configurações de estimativa
border_windowing_method = "blackman_harris" # Aplica o escurecimento nas bordas das imagens (Padrão: True, Valores Possíveis: True, False)
phase_windowing = None                                # Aplica o janelamento no sinal final da fase (Padrão: None, Valores Possíveis: None, "initial", "central")
