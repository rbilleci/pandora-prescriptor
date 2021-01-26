find * -not -path "logs" -not -path "*.log" -not -path "predictions*" -not -path "prescriptions*" -not -path bootstrap.sh -not -path "sandbox_*" -not -path work.tar.gz -delete
tar xvfz work.tar.gz
