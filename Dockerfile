FROM ubuntu:18.04

RUN apt update -y
RUN apt install --no-install-recommends -y \
        build-essential cmake
RUN apt install --no-install-recommends -y \
        ca-certificates git sudo \
        libssl-dev libz-dev libuv1-dev
RUN apt autoremove -y

# Add uWebSockets
WORKDIR /app
ADD ./install-ubuntu.sh /app/
RUN ./install-ubuntu.sh

# Add remaining code
WORKDIR /app/build
ADD ./vendor /app/vendor
ADD ./src /app/src
ADD ./CMakeLists.txt /app/CMakeLists.txt

# Build
RUN cmake ..
RUN make -j8