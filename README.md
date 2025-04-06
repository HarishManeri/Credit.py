import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from typing import Dict, Tuple, Optional
from enum import Enum

# Your existing classes and methods here
class TransactionRisk(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class AdvancedFraudDetection:
    def __init__(self):
        self.user_db = {}
        self.suspicious_ips = set()

    def register_user(self, username: str, password: str, card_number: str, cvv: str, expiry_date: str,
                      biometric_data: Optional[Dict[str, bytes]] = None,
                      device_info: Optional[Dict[str, str]] = None,
                      security_questions: Optional[Dict[str, str]] = None):
        self.user_db[username] = {
            "password": password,
            "card_info": {
                "card_number": card_number,
                "cvv": cvv,
                "expiry_date": expiry_date
            },
            "biometric_data": biometric_data,
            "trusted_devices": {self._generate_device_fingerprint(device_info)} if device_info else set(),
            "trusted_locations": set(),
            "security_questions": security_questions or {}
        }

    def _generate_device_fingerprint(self, device_info: Dict[str, str]) -> str:
        return str(hash(frozenset(device_info.items())))

class FraudDetectionMLModel:
    def __init__(self):
        self.model = None
        self.features = ['amount', 'device_fingerprint', 'ip_risk', 'location_change', 'biometric_match']

    def generate_sample_data(self, num_samples: int = 5000):
        np.random.seed(42)
        data = {
            'amount': np.random.uniform(10, 10000, num_samples),
            'device_fingerprint': np.random.choice([0, 1], num_samples),
            'ip_risk': np.random.choice([0, 1], num_samples),
            'location_change': np.random.choice([0, 1], num_samples),
            'biometric_match': np.random.choice([0, 1], num_samples),
            'is_fraud': np.random.choice([0, 1], num_samples, p=[0.9, 0.1])
        }
        return pd.DataFrame(data)

    def train_model(self, data: pd.DataFrame):
        X = data[self.features]
        y = data['is_fraud']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, "fraud_detection_model.pkl")

    def predict_transaction(self, transaction: dict) -> bool:
        if not self.model:
            self.model = joblib.load("fraud_detection_model.pkl")
        transaction_df = pd.DataFrame([transaction])
        prediction = self.model.predict(transaction_df)[0]
        return bool(prediction)

class EnhancedFraudDetection(AdvancedFraudDetection):
    def __init__(self):
        super().__init__()
        self.ml_model = FraudDetectionMLModel()
        self.ml_model.train_model(self.ml_model.generate_sample_data())

    def verify_transaction_with_ml(self, username: str, amount: float, location: str,
                                    ip_address: str, device_info: Dict[str, str],
                                    biometric_data: Optional[Dict[str, bytes]] = None) -> Tuple[bool, str, TransactionRisk]:
        if username not in self.user_db:
            return False, "User  not found", TransactionRisk.CRITICAL
        device_fingerprint = self._generate_device_fingerprint(device_info)
        biometric_match = 1 if biometric_data else 0
        transaction_features = {
            'amount': amount,
            'device_fingerprint': int(device_fingerprint in self.user_db[username]['trusted_devices']),
            'ip_risk': int(ip_address in self.suspicious_ips),
            'location_change': int(location not in self.user_db[username]['trusted_locations']),
            'biometric_match': biometric_match
        }
        is_fraud = self.ml_model.predict_transaction(transaction_features)
        if is_fraud:
            return False, "Fraudulent transaction detected by ML model", TransactionRisk.CRITICAL
        return True, "Transaction approved", TransactionRisk.LOW

# Streamlit app
def main():
    st.title("Fraud Detection System")

    # Create an instance of the EnhancedFraudDetection class
    fraud_detection = EnhancedFraudDetection()

    # Create tabs for user registration and transaction verification
    tab1, tab2 = st.tabs(["User  Registration", "Transaction Verification"])

    with tab1:
        st.header("Register User")
        username = st.text_input("Enter username:")
        password = st.text_input("Enter password:", type="password")
        card_number = st.text_input("Enter card number:")
        cvv = st.text_input("Enter CVV:", type="password")
        expiry_date = st.text_input("Enter expiry date (MM/YY):")
        device_info = {"OS": st.text_input("Device OS:"), "Browser": st.text_input("Browser:")}
        security_question = st.text_input("Security Question (e.g., First pet name):")
        security_answer = st.text_input("Answer:")

        if st.button("Register User"):
            fraud_detection.register_user(
                username=username,
                password=password,
                card_number=card_number,
                cvv=cvv,
                expiry_date=expiry_date,
                device_info=device_info,
                security_questions={security_question: security_answer}
            )
            st.success("User  registered successfully!")

    with tab2:
        st.header("Verify Transaction")
        username = st.text_input("Enter username for transaction verification:")
        amount = st.number_input("Enter transaction amount:", min_value=0.0, format="%.2f")
        location = st.text_input("Enter transaction location:")
        ip_address = st.text_input("Enter IP address:")
        device_info = {"OS": st.text_input("Device OS for transaction:"), "Browser": st.text_input("Browser for transaction:")}
        biometric_data = st.checkbox("Biometric data available?")

        if st.button("Verify Transaction"):
            is_valid, message, risk_level = fraud_detection.verify_transaction_with_ml(
                username=username,
                amount=amount,
                location=location,
                ip_address=ip_address,
                device_info=device_info,
                biometric_data={"data": b"biometric_data"} if biometric_data else None
            )
            st.write(f"Valid: {is_valid}")
            st.write(f"Message: {message}")
            st.write(f"Risk Level: {risk_level.value}")

if __name__ == "__main__":
    main()
