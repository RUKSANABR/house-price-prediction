import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ToastAndroid } from 'react-native';
import { Avatar, Button, Card } from '@rneui/base';
import firestore from '@react-native-firebase/firestore';

export default function Notification({ route }) {
  const { userD } = route.params;
  const [data, setData] = useState([]);
  const [isUpdated, setIsUpdated] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const propertyRef = firestore().collection('contract');
        const query1 = propertyRef.where('customerId', '==', userD.email);
        const query2 = propertyRef.where('contracterId', '==', userD.email);

        Promise.all([query1.get(), query2.get()])
          .then((snapshots) => {
            const dataArray = [];

            snapshots.forEach((snapshot) => {
              snapshot.forEach((doc) => {
                dataArray.push({ contractId: doc.id, ...doc.data() });
              });
            });

            // Now, dataArray contains the results of both queries
            // You can filter or process this data as needed
            setData(dataArray);
          })
          .catch((error) => {
            console.error('Error getting documents: ', error);
          });
      } catch (error) {
        console.error('An error occurred: ', error);
      }
    };

    fetchData();
  }, [isUpdated]); 

  const handleAccept = async (contractId,propertyId) => {
    const bookingRef = firestore().collection('contract').doc(contractId);
    const propertyRef= firestore().collection('property').doc(propertyId);
    try {
      await bookingRef.update({ status: 'Accepted' });
      await propertyRef.update({ status: 'Accepted' });
      console.log('Booking status updated to accepted');
      ToastAndroid.show('Requested status updated to accepted', ToastAndroid.SHORT);
      setIsUpdated(true); 
    } catch (error) {
      console.error('Error updating booking status: ', error);
    }
  };

  const handleDecline = async (contractId) => {
    const bookingRef = firestore().collection('contract').doc(contractId);
    try {
      await bookingRef.update({ status: 'Declined' });
      console.log('Requested status updated to declined');
      ToastAndroid.show('Booking status updated to declined', ToastAndroid.SHORT);
      setIsUpdated(true); 
    } catch (error) {
      console.error('Error updating booking status: ', error);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Notifications</Text>
      {userD.userType === 'user' ? (
        data
          .filter((item) => item.status === 'Requested')
          .map((item, index) => (
            <Card key={index} containerStyle={styles.cards}>
              <View style={styles.rows}>
                <Avatar
                  source={require('../assets/profile.png')}
                  size={50}
                  containerStyle={{ marginRight: 10 }}
                />
                <Text style={styles.font1}>Contractor: </Text>
                <Text style={[styles.font1, { color: 'black', fontWeight: 'bold' }]}>{item.contracterName}</Text>
              </View>
              <View style={styles.rows}>
                <Text style={styles.font2}>Requested to take the contract at the price</Text>
                <Text style={[styles.font2, { color: 'black', fontWeight: 'bold' }]}>:₹{item.contracterPrice}</Text>
              </View>
              <View style={[styles.rows, { justifyContent: 'space-around', marginTop: 10 }]}>
                <Button
                  title="Decline"
                  color="#000000"
                  containerStyle={styles.button}
                  titleStyle={{ fontSize: 18 }}
                  onPress={() => handleDecline(item.contractId)}
                />
                <Button
                  title="Accept"
                  color="#52A9E3"
                  containerStyle={styles.button}
                  titleStyle={{ fontSize: 18 }}
                  onPress={() => handleAccept(item.contractId,item.propertyId)}
                />
              </View>
            </Card>
          ))
      ) : (
        data
          .filter((item) => item.status === 'Accepted')
          .map((item, index) => (
            <Card key={index} containerStyle={[styles.cards, { height: 120 }]}>
              <View style={styles.rows}>
                <Avatar
                  source={require('../assets/profile.png')}
                  size={50}
                  containerStyle={{ marginRight: 10 }}
                />
                <Text style={styles.font1}>Customer: </Text>
                <Text style={styles.font1}>{item.customerName}</Text>
              </View>
              <View style={styles.rows}>
                <Text style={styles.font2}>Accepted Your Request at the price</Text>
                <Text style={[styles.font2, { color: 'black', fontWeight: 'bold' }]}>:₹{item.contracterPrice}</Text>
              </View>
            </Card>
          ))
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 10,
    marginTop: 20,
  },
  title: {
    fontSize: 25,
    color: 'black',
    marginLeft: 20,
    fontWeight: 'bold',
  },
  cards: {
    width: '95%',
    height: 160,
    borderRadius: 20,
    alignSelf: 'center',
  },
  font1: {
    textAlign: 'justify',
    fontSize: 20,
    color: 'black',
  },
  font2: {
    textAlign: 'justify',
    fontSize: 15,
    marginTop: 5,
    color: 'black',
  },
  rows: {
    flexDirection: 'row',
    alignItems: 'center',
    textAlign:'justify'
  },
  button: {
    width: 150,
    borderRadius: 10,
  },
});
