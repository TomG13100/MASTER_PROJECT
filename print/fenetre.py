import sys
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt


class ResultWindow(QWidget):
    def __init__(self, action, precision):
        super().__init__()

        # Configurer la fenêtre
        self.setWindowTitle("Résultat de l'IA")
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint
        )  # Toujours au premier plan, sans bordures
        self.setWindowModality(Qt.ApplicationModal)  # Bloque les interactions ailleurs

        # Taille fixe et centrée
        self.setFixedSize(400, 200)
        self.center_window()

        # Contenu de la fenêtre
        label_action = QLabel(f"Prédiction : {action}")
        label_action.setAlignment(Qt.AlignCenter)
        label_action.setStyleSheet("font-size: 18px; font-weight: bold; color: #444;")

        label_precision = QLabel(f"Précision : {precision:.2f}%")
        label_precision.setAlignment(Qt.AlignCenter)
        label_precision.setStyleSheet(self.get_precision_color(precision))

        # Bouton OK
        button_ok = QPushButton("OK")
        button_ok.clicked.connect(self.close_window)
        button_ok.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        # Agencement
        layout = QVBoxLayout()
        layout.addWidget(label_action)
        layout.addWidget(label_precision)
        layout.addWidget(button_ok)
        self.setLayout(layout)

        # Appliquer des styles globaux à la fenêtre
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 12px;
                padding: 15px;
            }
        """)

    def get_precision_color(self, precision):
        """Renvoie la couleur CSS basée sur la précision."""
        if precision > 75:
            color = "#FF0000"  # Rouge
        elif precision > 50:
            color = "#FFA500"  # Orange
        elif precision > 25:
            color = "#0000FF"  # Bleu
        else:
            color = "#008000"  # Vert

        return f"font-size: 16px; font-weight: bold; color: {color};"

    def close_window(self):
        """Ferme la fenêtre lorsque le bouton OK est cliqué."""
        self.close()

    def center_window(self):
        """Centre la fenêtre sur l'écran."""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.geometry()
            x = (screen_geometry.width() - self.width()) // 2
            y = (screen_geometry.height() - self.height()) // 2
            self.move(x, y)
        else:
            self.move(100, 100)  # Position par défaut si aucune géométrie n'est disponible


if __name__ == "__main__":
    # Crée l'application
    app = QApplication(sys.argv)

    # Exemple d'affichage
    action = "Envoyer SMUR"
    precision = 95

    # Instancie la fenêtre
    window = ResultWindow(action, precision)
    window.show()

    # Boucle principale
    sys.exit(app.exec_())
