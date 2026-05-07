#!/bin/bash
# Lokal TLS sertifikaları — CA + broker + (opsiyonel) istemci
# Çalıştır: bash certs/generate_certs.sh <BROKER_IP>
# Örnek:    bash certs/generate_certs.sh 192.168.1.100
#
# Üretilen dosyalar:
#   certs/ca.key        — CA özel anahtarı (güvenli tut, paylaşma)
#   certs/ca.crt        — CA sertifikası   (Jetson'a kopyalanır)
#   certs/broker.key    — Broker özel anahtarı
#   certs/broker.crt    — Broker sertifikası (Mosquitto'ya mount edilir)

set -e

BROKER_IP="${1:?Kullanım: $0 <BROKER_IP>}"
DIR="$(cd "$(dirname "$0")" && pwd)"

echo ">>> Sertifikalar $DIR dizinine yazılıyor (broker IP: $BROKER_IP)"

# --- CA ---
openssl genrsa -out "$DIR/ca.key" 2048
openssl req -new -x509 -days 3650 -key "$DIR/ca.key" -out "$DIR/ca.crt" \
  -subj "/CN=WindTurbineCA/O=WindTurbine/C=TR"

# --- Broker (SAN ile — IP adresi için zorunlu) ---
openssl genrsa -out "$DIR/broker.key" 2048
openssl req -new -key "$DIR/broker.key" -out "$DIR/broker.csr" \
  -subj "/CN=$BROKER_IP/O=WindTurbine/C=TR"

openssl x509 -req -days 3650 \
  -in "$DIR/broker.csr" \
  -CA "$DIR/ca.crt" -CAkey "$DIR/ca.key" -CAcreateserial \
  -extfile <(printf "subjectAltName=IP:%s" "$BROKER_IP") \
  -out "$DIR/broker.crt"

rm -f "$DIR/broker.csr" "$DIR/ca.srl"

echo ""
echo "✓ Tamamlandı. Dosyalar:"
ls -1 "$DIR"/*.{crt,key} 2>/dev/null
echo ""
echo "Jetson'a kopyalamak için:"
echo "  scp $DIR/ca.crt jetson@<JETSON_IP>:/home/jetson/certs/ca.crt"
