ARG BUILD_ID
FROM tensorcomprehensions/linux-xenial-gcc5-tapir5.0:${BUILD_ID}

ARG BUILD
ADD ./common/install_cuda.sh install_cuda.sh
RUN bash ./install_cuda.sh

CMD ["bash"]
